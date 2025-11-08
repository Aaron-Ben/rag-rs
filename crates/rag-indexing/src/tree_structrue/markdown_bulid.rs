use crate::tree_structrue::{Node, NodeId, NodeTree};
use pulldown_cmark::{Parser, Options, Event, Tag};
use anyhow::Result;
use std::fmt;


pub struct MarkdownParser {
    document_id: String,
    file_name: Option<String>,
}

impl MarkdownParser {
    pub fn new(document_id: String, file_name: Option<String>) -> Self {
        Self { document_id, file_name }
    }

    pub fn parse(&self, content: &str) -> Result<NodeTree> {
        let options = Options::all();
        let parser = Parser::new_ext(content, options);

        let mut tree = NodeTree::new(Node::new_root(
            self.document_id.clone(),
            self.file_name.clone(),
        ));

        let root_id = tree.root;

        // 标题栈：(node_id, hierarchy_vec)
        let mut heading_stack: Vec<(NodeId, Vec<String>)> = vec![(root_id, vec!["Root".to_string()])];
        let mut current_parent_id = root_id;
        let mut current_hierarchy = vec!["Root".to_string()];

        // 状态标志
        let mut in_code_block = false;
        let mut in_table = false;
        let mut in_image = false;

        // 缓冲区
        let mut table_header: Option<Vec<String>> = None;
        let mut table_buffer: Vec<Vec<String>> = Vec::new();
        let mut current_row: Vec<String> = vec![];
        let mut code_buffer = String::new();
        let mut paragraph_buffer = String::new();

        let mut image_alt = String::new();
        let mut image_path = String::new();

        // 待处理的标题
        struct PendingHeading {
            level: u32,
            text: String,
            _parent_id: NodeId,
            _parent_hierarchy: Vec<String>,
        }
        let mut pending_heading: Option<PendingHeading> = None;

        // 全局 chunk 计数
        let mut chunk_index = 0;

        for event in parser {
            match event {
                // === 开始标签 ===
                Event::Start(tag) => {
                    match tag {
                        Tag::Heading { level, .. } => {
                            // 弹出超出当前级别的栈顶
                            while heading_stack.len() > level as usize {
                                heading_stack.pop();
                            }

                            let (parent_id, parent_hier) = heading_stack.last().cloned()
                                .unwrap_or((root_id, vec!["Root".to_string()]));

                            pending_heading = Some(PendingHeading {
                                level: level as u32,
                                text: String::new(),
                                _parent_id: parent_id,
                                _parent_hierarchy: parent_hier,
                            });
                        }

                        Tag::CodeBlock(_) => {
                            in_code_block = true;
                            code_buffer.clear();
                        }

                        Tag::Table(_) => {
                            in_table = true;
                            table_header = None;
                            table_buffer.clear();
                            current_row.clear();
                        }

                        Tag::Image { dest_url, title, .. } => {
                            in_image = true;
                            image_alt = title.to_string();
                            image_path = dest_url.to_string();
                        }

                        _ => {}
                    }
                }

                // === 结束标签 ===
                Event::End(tag_end) => {
                    match tag_end {
                        pulldown_cmark::TagEnd::Heading(_) => {
                            if let Some(heading) = pending_heading.take() {
                                let title = heading.text.trim();
                                if title.is_empty() {
                                    continue; // 跳过空标题
                                }
                                let title_str = title.to_string();

                                // 确保栈深度正确
                                while heading_stack.len() > heading.level as usize {
                                    heading_stack.pop();
                                }

                                let (parent_id, parent_hier) = heading_stack.last().cloned()
                                    .unwrap_or((root_id, vec!["Root".to_string()]));

                                let mut new_hier = parent_hier.clone();
                                new_hier.push(title_str.clone());

                                let intermediate = Node::new_intermediate(
                                    parent_id,
                                    Some(title_str.clone()),
                                    new_hier.clone(),
                                    self.document_id.clone(),
                                );
                                let new_id = intermediate.id();
                                tree.add_node(intermediate)?;

                                // 入栈
                                heading_stack.push((new_id, new_hier.clone()));

                                // 更新当前上下文
                                current_parent_id = new_id;
                                current_hierarchy = new_hier;
                            }
                        }

                        pulldown_cmark::TagEnd::Paragraph => {
                            if !paragraph_buffer.trim().is_empty() {
                                let text = paragraph_buffer.trim().to_string();
                                let leaf = Node::new_leaf(
                                    current_parent_id,
                                    text.clone(),
                                    text.len(),
                                    chunk_index,
                                    current_hierarchy.clone(),
                                    self.document_id.clone(),
                                    None,
                                    None,
                                    None,
                                    self.file_name.clone(),
                                );
                                tree.add_node(leaf)?;
                                chunk_index += 1;
                            }
                            paragraph_buffer.clear();
                        }

                        pulldown_cmark::TagEnd::CodeBlock => {
                            if in_code_block {
                                let text = code_buffer.trim_end().to_string();
                                if !text.is_empty() {
                                    let leaf = Node::new_leaf(
                                        current_parent_id,
                                        text.clone(),
                                        text.len(),
                                        chunk_index,
                                        current_hierarchy.clone(),
                                        self.document_id.clone(),
                                        None,
                                        None,
                                        None,
                                        self.file_name.clone(),
                                    );
                                    tree.add_node(leaf)?;
                                    chunk_index += 1;
                                }
                                in_code_block = false;
                                code_buffer.clear();
                            }
                        }

                        pulldown_cmark::TagEnd::TableHead => {
                            if in_table {
                                table_header = Some(current_row.clone());
                                current_row.clear();
                            }
                        }

                        pulldown_cmark::TagEnd::TableRow => {
                            if in_table && table_header.is_some() {
                                table_buffer.push(current_row.clone());
                                current_row.clear();
                            }
                        }

                        pulldown_cmark::TagEnd::Table => {
                            if in_table {
                                let mut markdown = String::new();
                                if let Some(header) = &table_header {
                                    markdown.push_str(&format!("| {} |\n", header.join(" | ")));
                                    markdown.push_str(&format!("| {} |\n", "--- | ".repeat(header.len()).trim_end()));
                                }
                                for row in &table_buffer {
                                    markdown.push_str(&format!("| {} |\n", row.join(" | ")));
                                }

                                if !markdown.trim().is_empty() {
                                    let mut table_hier = current_hierarchy.clone();
                                    table_hier.push(format!("table_{}", chunk_index));

                                    let leaf = Node::new_leaf(
                                        current_parent_id,
                                        markdown.clone(),
                                        markdown.len(),
                                        chunk_index,
                                        table_hier,
                                        self.document_id.clone(),
                                        None,
                                        None,
                                        None,
                                        self.file_name.clone(),
                                    );
                                    tree.add_node(leaf)?;
                                    chunk_index += 1;
                                }

                                table_header = None;
                                table_buffer.clear();
                                in_table = false;
                            }
                        }

                        pulldown_cmark::TagEnd::Image => {
                            if in_image {
                                let markdown = format!("![{}]({})", image_alt, image_path);
                                let mut img_hier = current_hierarchy.clone();
                                img_hier.push(format!("img_{}", chunk_index));

                                let image_id = image_path.split("/").last().unwrap_or("").to_string();

                                let leaf = Node::new_leaf(
                                    current_parent_id,
                                    markdown.clone(),
                                    markdown.len(),
                                    chunk_index,
                                    img_hier,
                                    self.document_id.clone(),
                                    if image_alt.is_empty() { None } else { Some(image_alt.clone()) },
                                    Some(image_path.clone()),
                                    Some(image_id),
                                    self.file_name.clone(),
                                );
                                tree.add_node(leaf)?;
                                chunk_index += 1;

                                in_image = false;
                                image_alt.clear();
                                image_path.clear();
                                paragraph_buffer.clear();
                            }
                        }

                        _ => {}
                    }
                }

                // === 文本事件 ===
                Event::Text(text) => {
                    let s = text.as_ref();

                    if let Some(heading) = &mut pending_heading {
                        heading.text.push_str(s);
                    } else if in_code_block {
                        code_buffer.push_str(s);
                        code_buffer.push('\n');
                    } else if in_table {
                        current_row.push(s.to_string());
                    } else if in_image {
                        image_alt.push_str(s);
                    } else if !s.trim().is_empty() {
                        paragraph_buffer.push_str(s);
                        paragraph_buffer.push(' ');
                    }
                }

                Event::Code(text) => {
                    if pending_heading.is_none() && !in_code_block {
                        paragraph_buffer.push_str(&format!("`{}` ", text));
                    }
                }

                Event::SoftBreak | Event::HardBreak => {
                    if !paragraph_buffer.is_empty() && pending_heading.is_none() && !in_table {
                        paragraph_buffer.push(' ');
                    }
                }

                _ => {}
            }
        }

        // 处理最后未结束的段落
        if !paragraph_buffer.trim().is_empty() {
            let text = paragraph_buffer.trim().to_string();
            let leaf = Node::new_leaf(
                current_parent_id,
                text.clone(),
                text.len(),
                chunk_index,
                current_hierarchy.clone(),
                self.document_id.clone(),
                None,
                None,
                None,
                self.file_name.clone(),
            );
            tree.add_node(leaf)?;
        }

        Ok(tree)
    }
}

// 添加 Display trait 的实现
impl fmt::Display for Node {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Node::Root(root) => {
                write!(f, "📁 ROOT: {} (文件: {:?})", 
                    root.document_id, 
                    root.metadata.file_name
                )
            }
            Node::Intermediate(inter) => {
                if let Some(title) = &inter.title {
                    write!(f, "📂 {}", title)
                } else {
                    write!(f, "📂 (未命名中间节点)")
                }
            }
            Node::Leaf(leaf) => {
                // 截取文本前500个字符显示
                let display_text = if leaf.text.chars().count() > 500 {
                    let truncated: String = leaf.text.chars().take(500).collect();
                    format!("{}...", truncated)
                } else {
                    leaf.text.clone()
                };
                // 移除换行符以便在一行显示
                let clean_text = display_text.replace('\n', " ").replace('\r', "");
                write!(f, "📄 {}", clean_text)
            }
        }
    }
}

// 为 NodeTree 实现 Display trait，实现分层打印
impl fmt::Display for NodeTree {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "🌳 文档树结构:")?;
        writeln!(f, "{}", "=".repeat(60))?;
        
        // 从根节点开始递归打印
        self.print_node_recursive(f, self.root, 0)?;
        
        writeln!(f, "{}", "=".repeat(60))?;
        writeln!(f, "📊 统计信息:")?;
        writeln!(f, "   总节点数: {}", self.nodes.len())?;
        
        // 统计各类节点数量
        let mut root_count = 0;
        let mut intermediate_count = 0;
        let mut leaf_count = 0;
        
        for node in self.nodes.values() {
            match node {
                Node::Root(_) => root_count += 1,
                Node::Intermediate(_) => intermediate_count += 1,
                Node::Leaf(_) => leaf_count += 1,
            }
        }
        
        writeln!(f, "   根节点: {}", root_count)?;
        writeln!(f, "   中间节点: {}", intermediate_count)?;
        writeln!(f, "   叶子节点: {}", leaf_count)?;
        
        Ok(())
    }
}

// 递归打印节点
impl NodeTree {
    fn print_node_recursive(&self, f: &mut fmt::Formatter, node_id: NodeId, depth: usize) -> fmt::Result {
        if let Some(node) = self.nodes.get(&node_id) {
            // 打印缩进
            let indent = "  ".repeat(depth);
            
            // 根据节点类型选择图标和内容
            let (icon, content) = match node {
                Node::Root(root) => {
                    let file_info = match &root.metadata.file_name {
                        Some(name) => format!(" [{}]", name),
                        None => "".to_string(),
                    };
                    ("🌳", format!("ROOT{}{}", root.document_id, file_info))
                }
                Node::Intermediate(inter) => {
                    let title = inter.title.as_deref().unwrap_or("(未命名)");
                    let hierarchy_path = &inter.metadata.hierarchy;
                    let path = hierarchy_path.join(" > ");
                    ("📂", format!("{} (路径: {})", title, path))
                }
                Node::Leaf(leaf) => {
                    let is_image = leaf.metadata.image_path.is_some();
                    let icon = if is_image { "🖼️" } else { "📄" };

                    let chunk_info = leaf.metadata.chunk_size
                        .map(|size| format!("[chuck_{}]",size))
                        .unwrap_or("[chunk]".to_string());

                    let content = if is_image {
                        let alt = leaf.metadata.image_alt.as_deref().unwrap_or("无描述");
                        let path = leaf.metadata.image_path.as_deref().unwrap_or("未知路径");
                        let image_id = leaf.metadata.image_id.as_deref().unwrap_or("");

                        // 只显示语义信息，不显示完整 Markdown
                        if !image_id.is_empty() {
                            format!("{} [{}] {} -> {}", chunk_info, alt, image_id, path)
                        } else {
                            format!("{} [{}] {}", chunk_info, alt, path)
                        }
                    } else {
                        let display_text = if leaf.text.chars().count() > 500 {
                            let truncated: String = leaf.text.chars().take(500).collect();
                            format!("{}...", truncated)
                        } else {
                            leaf.text.clone()
                        };

                        let clean_text = display_text.replace('\n', " ").replace('\r', "");
                        format!("{} {}", chunk_info, clean_text)
                    };

                    (icon, content)
                }
            
            };
            
            writeln!(f, "{}{} {}", indent, icon, content)?;
            
            // 如果有子节点，递归打印
            if !node.children().is_empty() {
                for &child_id in node.children() {
                    self.print_node_recursive(f, child_id, depth + 1)?;
                }
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests { 
    use super::*;
    use anyhow::Result;
    use serde_json;

    
    const TEST_MARKDOWN: &str = r#"
# ChatGPT出现以来中美大模型发展报告
## 概述
自2022年11月30日OpenAI发布ChatGPT以来，全球人工智能领域发生了前所未有的变革。ChatGPT的发布标志着大语言模型时代的正式开启，激发了中美两国在AI领域的激烈竞争。这场始于技术突破的竞争，已经演变为涉及国家战略、产业生态、人才储备、基础设施等多个维度的全面博弈。
### 历史背景与意义

ChatGPT的出现并非偶然，而是人工智能发展到一定阶段的必然产物。在其发布之前，人工智能已经历了多次高潮与低谷：1950年代的符号主义、1980年代的专家系统、1990年代的机器学习兴起、2010年代的深度学习革命。每一次技术浪潮都推动着人工智能向更智能、更实用的方向发展，而ChatGPT所代表的大语言模型技术，则是这一历史进程的集大成者。

大语言模型的兴起有三大关键技术背景：首先，Transformer架构的提出为模型扩展提供了理论基础；其次，互联网上海量文本数据为模型训练提供了丰富素材；最后，GPU等并行计算设备的普及为大规模模型训练提供了硬件支撑。这些因素的结合，使得构建具有通用智能能力的大模型成为可能。
### 中美竞争的深层逻辑

中美两国在AI大模型领域的竞争，本质上是两种发展模式、两种技术路线、两种价值理念的较量。美国以硅谷为中心，代表了以企业为主导、市场驱动的创新模式，强调技术开源、生态开放、商业化优先。而中国则以北京、上海、深圳为核心，形成了以政府引导、产业协同、应用导向的发展模式，更加注重技术的产业化落地和社会效益。

这种竞争不仅体现在技术研发上，更深层次地反映了未来数字经济的主导权、人工智能标准制定的话语权、以及科技创新的主导地位谁能掌握。胜者将在未来几十年的全球科技竞争中占据有利位置，败者可能面临技术依赖和产业边缘化的风险。
### 当前发展态势

根据斯坦福大学人工智能研究所（HAI）发布的《2025年人工智能指数报告》，中美顶级AI大模型的性能差距已从2023年的17.5%大幅缩至0.3%，几乎实现技术平价。这一里程碑式的进展标志着全球AI竞争格局的深刻变化：从美国的绝对领先转向中美双强并立。

这一变化具有多重意义：首先，中国在AI技术领域的快速追赶证明了其技术创新能力的显著提升；其次，技术差距的缩小意味着未来竞争将更加激烈，任何微小的技术突破都可能改变竞争格局；最后，这种变化也促使两国在AI治理、伦理标准、国际合作等方面寻求新的平衡。
        
## 历史发展时间线

### 技术萌芽期：通往大语言模型的之路（2017-2020）

在ChatGPT正式发布之前，人工智能领域已经为大语言模型的到来做了充分的技术准备。2017年，Google研究团队发表的《Attention Is All You Need》论文，提出了革命性的Transformer架构，为后续大语言模型的发展奠定了理论基础。Transformer架构的注意力机制解决了传统循环神经网络在处理长序列时的梯度消失问题，为构建能够处理大规模文本数据的人工智能模型提供了可能。

这一时期的技术探索为后续爆发式发展积蓄了力量，但受限于算力成本、数据规模和技术成熟度，大语言模型仍处于实验室阶段，主要服务于研究目的和特定任务，应用范围相对有限。

### ChatGPT发布前的准备阶段（2020-2022）
| 时间 | 事件 | 影响 | 技术意义 |
|------|------|------|----------|
| 2020年5月 | OpenAI发布GPT-3 | 1750亿参数，展现大模型潜力 | 首次验证了规模扩展的效果 |
| 2021年 | 谷歌发布LaMDA | 对话AI技术突破 | 专门针对对话场景优化 |
| 2022年4月 | OpenAI发布DALL-E 2 | 多模态AI能力展示 | 文本到图像生成技术突破 |
| 2022年7月 | Google发布PaLM | 5400亿参数大模型 | 证明了更大规模模型的潜力 |
| 2022年8月 | Meta发布LLaMA | 开源模型系列启动 | 为开源社区奠定基础 |

**2020年5月：GPT-3的震撼登场**
GPT-3的发布标志着大语言模型从概念走向实用的重要转折点。这个拥有1750亿参数的模型展现出了惊人的能力：能够进行流畅的对话、创作诗歌、编写代码、翻译文本，甚至展现出一定程度的推理能力。GPT-3的成功证明了"规模扩展"（Scale）——通过增加模型参数和训练数据来提升模型性能——这一路线的可行性。

然而，GPT-3也暴露了早期大模型的问题：虽然能力强大，但仍需要精细的提示工程才能发挥最佳效果；模型输出存在不稳定性；特别是在处理专业领域问题时表现不够可靠。尽管如此，GPT-3的成功为整个行业指明了方向，也为ChatGPT的诞生做好了技术准备。

**2021-2022年：多路径探索与竞争加速**

在GPT-3之后，谷歌、Meta等科技巨头纷纷加入大模型竞赛。谷歌发布的LaMDA（Language Model for Dialogue Applications）专门针对对话场景进行了优化，展现出更好的对话连贯性和上下文理解能力。同期，DALL-E 2的发布则展示了多模态AI的巨大潜力，能够根据文字描述生成高质量图像。

Meta的LLaMA模型系列则选择了开源路线，虽然最初只对学术机构开放，但其开源策略为后续整个生态的发展奠定了基础。开源路线的选择在后来证明极具前瞻性，为中国等国家的AI发展提供了重要参考。

**2023年：中国大模型元年**

2023年被称为中国大模型元年。从3月百度文心一言首次发布开始，中国各大科技公司纷纷在6个月内推出自己的大模型产品，形成了"百模大战"的壮观场面。这种集中爆发的现象反映了中国AI产业的整体实力和快速响应能力。

中国大模型的发展呈现出与美国不同的特点：更加注重实用性和本地化，从设计阶段就考虑中文语境、法律法规、文化背景等因素。同时，中国企业在大模型的应用落地上展现出了更强的执行力，短时间内将AI能力集成到各种产品和服务中。

**2024-2025年：技术差距缩小与竞争加剧**

进入2024年后，中美AI技术差距开始显著缩小。OpenAI发布的o1系列在推理能力上取得重大突破，中国各大厂商的模型在综合能力上快速追赶。到2025年，双方在多个技术维度上已经基本持平，标志着AI竞争进入新阶段。

这一阶段的竞争焦点从单纯的技术指标转向应用场景、用户体验、生态建设等综合实力。两国都在寻找自己的差异化优势：美国继续在基础技术和产品生态上发力，中国则在中文本土化和产业化应用上深耕细作。

![AI芯片算力对比](/Users/xuenai/Code/rag-rs/docs/imgs/ai_chips_3.jpg)

```python
print("hello world")
```
    "#;


    #[test]
    fn test1() -> Result<()> {

        let parser = MarkdownParser::new("doc-001".to_string(), Some("rag.md".to_string()));
        let tree = parser.parse(TEST_MARKDOWN)?;
        
        println!("=== 树形结构显示 ===");
        println!("{}", tree);
        
        Ok(())
    }

    #[test]
    fn test2() -> Result<()> {

        let parser = MarkdownParser::new("doc-002".to_string(), Some("rag_report.md".to_string()));
        let tree = parser.parse(TEST_MARKDOWN)?;
        let json = serde_json::to_string_pretty(&tree)?;
        println!("{}", json);
        Ok(())
    }

}
