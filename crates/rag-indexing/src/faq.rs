use serde::{Deserialize, Serialize};
use std::fmt;
use crate::tiktoken::count_tokens;
use jieba_rs::Jieba;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct FAQEntry {
    pub category: String,
    pub q: String,
    pub a: String,
    pub tags: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct FAQChunk {
    pub chunk_id: String,
    pub faq_id: String,
    pub category: String,
    pub title: String,           // 标题
    pub content: String,         // Q + A
    pub tags: Vec<String>,
    pub token_count: usize,
}

pub struct FAQChunker {
    max_tokens: usize,
    overlap: usize,
    model: String,
    jieba: Jieba,
}

impl FAQChunker {
    /// 创建新的 FAQChunker
    /// 
    /// # 参数
    /// - `max_tokens`: 每个 chunk 的最大 token 数
    /// - `overlap`: 重叠的句子数（用于超长 QA 拆分）
    /// - `model`: 模型名称，用于 tokenizer（如 "qwen-max", "gpt-4o"）
    pub fn new(max_tokens: usize, overlap: usize, model: String) -> Self {
        Self { 
            max_tokens, 
            overlap,
            model,
            jieba: Jieba::new(),
        }
    }

    /// 使用模型原生的 tokenizer 计算 token 数
    fn count_tokens(&self, text: &str) -> usize {
        count_tokens(text, &self.model)
    }

    /// 按 QA 对分块（每个 QA 是一个 chunk，超长时拆分）
    pub fn chunk_by_qa(&self, entries: Vec<FAQEntry>) -> Vec<FAQChunk> {
        let mut chunks = Vec::new();

        for (entry_idx, entry) in entries.iter().enumerate() {
            // 生成 faq_id（按分类+序号命名，如 "退货申请类-001"）
            let category_key = entry.category.replace(" ", "-").to_lowercase();
            let faq_id = format!("faq-{}-{:03}", category_key, entry_idx + 1);
            
            // 构建 QA 完整内容
            let raw_content = format!("Q: {}\nA: {}", entry.q.trim(), entry.a.trim());
            let raw_token_count = self.count_tokens(&raw_content);

            // 处理超长 QA（拆分后生成多个 chunk）
            if raw_token_count > self.max_tokens {
                let split_chunks = self.split_long_qa(&raw_content, &faq_id, entry);
                chunks.extend(split_chunks);
            } else {
                // 正常长度：直接生成单个 chunk
                chunks.push(FAQChunk {
                    chunk_id: format!("{}-chunk-1", faq_id),
                    faq_id: faq_id.clone(),
                    category: entry.category.clone(),
                    title: entry.q.trim().to_string(),
                    content: raw_content,
                    tags: entry.tags.clone(),
                    token_count: raw_token_count,
                });
            }
        }

        chunks
    }

    /// 优化：超长 QA 拆分（保留上下文重叠，确保拆分后语义完整）
    fn split_long_qa(&self, text: &str, faq_id: &str, entry: &FAQEntry) -> Vec<FAQChunk> {
        let mut chunks = Vec::new();
        
        // 优先使用语义单元切分（对中文更友好）
        let semantic_units = self.split_into_semantic_units(text);
        
        // 如果没有语义单元或只有一个单元（可能全是英文），则回退到句子切分
        let units: Vec<String> = if semantic_units.len() > 1 {
            semantic_units
        } else {
            // 回退到句子切分
            text.split(&['。', '！', '？', '.', '!', '?', '；', ';'])
                .filter(|s| !s.trim().is_empty())
                .map(|s| s.trim().to_string())
                .collect()
        };

        let mut current_chunk_idx = 1;
        let mut current_units = Vec::new();
        let mut current_token_count = 0;

        for (unit_idx, unit) in units.iter().enumerate() {
            let unit_trimmed = unit.trim();
            if unit_trimmed.is_empty() { continue; }
            
            let unit_tokens = self.count_tokens(unit_trimmed);

            // 预判：添加当前单元后是否超 max_tokens
            let new_token_count = current_token_count + unit_tokens;
            if new_token_count > self.max_tokens && !current_units.is_empty() {
                // 生成当前 chunk
                let chunk_content = current_units.join("");
                chunks.push(FAQChunk {
                    chunk_id: format!("{}-chunk-{}", faq_id, current_chunk_idx),
                    faq_id: faq_id.to_string(),
                    category: entry.category.clone(),
                    title: entry.q.trim().to_string(),
                    content: chunk_content,
                    tags: entry.tags.clone(),
                    token_count: current_token_count,
                });

                // 重置当前单元（保留重叠部分，避免语义断裂）
                current_units.clear();
                if self.overlap > 0 {
                    // 从当前单元往前取 overlap 个单元作为重叠
                    let start_idx = if unit_idx >= self.overlap {
                        unit_idx - self.overlap
                    } else {
                        0
                    };
                    for u in &units[start_idx..=unit_idx] {
                        current_units.push(u.trim().to_string());
                    }
                    // 重新计算重叠后的 token 数
                    current_token_count = self.count_tokens(
                        &current_units.join("")
                    );
                } else {
                    current_units.push(unit_trimmed.to_string());
                    current_token_count = unit_tokens;
                }
                current_chunk_idx += 1;
            } else {
                current_units.push(unit_trimmed.to_string());
                current_token_count = new_token_count;
            }
        }

        // 添加最后一个 chunk
        if !current_units.is_empty() {
            let chunk_content = current_units.join("");
            chunks.push(FAQChunk {
                chunk_id: format!("{}-chunk-{}", faq_id, current_chunk_idx),
                faq_id: faq_id.to_string(),
                category: entry.category.clone(),
                title: entry.q.trim().to_string(),
                content: chunk_content,
                tags: entry.tags.clone(),
                token_count: current_token_count,
            });
        }

        chunks
    }

    /// 使用 jieba 分词将文本切分为语义单元
    /// 语义单元是在标点符号或自然停顿处切分的文本片段
    fn split_into_semantic_units(&self, text: &str) -> Vec<String> {
        // 使用 jieba 进行精确分词
        let words = self.jieba.cut(text, true);
        let mut units = Vec::new();
        let mut current_unit = String::new();

        // 定义语义终止符（标点符号）
        let sentence_endings = ['.', '。', '!', '！', '?', '？', ';', '；'];
        let clause_endings = ['，', ',', '、', '：', ':', '\n'];

        for word in words {
            current_unit.push_str(word);
            
            // 检查是否包含句子结束符
            let has_sentence_end = word.chars().any(|c| sentence_endings.contains(&c));
            // 检查是否包含分句符（在较长文本中）
            let has_clause_end = word.chars().any(|c| clause_endings.contains(&c));
            
            // 如果遇到句子结束符，切分单元
            if has_sentence_end {
                units.push(current_unit.trim().to_string());
                current_unit.clear();
            } else if has_clause_end && current_unit.len() > 50 {
                // 对于较长的文本，在分句符处也切分（但只在单元较长时）
                // 这样可以避免过度切分短文本
                units.push(current_unit.trim().to_string());
                current_unit.clear();
            }
        }

        // 添加最后一个单元
        if !current_unit.trim().is_empty() {
            units.push(current_unit.trim().to_string());
        }

        // 过滤空单元
        units.into_iter().filter(|u| !u.is_empty()).collect()
    }
}

// 实现 FAQChunk 的格式化输出（便于打印查看）
impl fmt::Display for FAQChunk {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== Chunk: {} ===", self.chunk_id)?;
        writeln!(f, "FAQ ID: {}", self.faq_id)?;
        writeln!(f, "分类: {}", self.category)?;
        writeln!(f, "标题: {}", self.title)?;
        writeln!(f, "内容: {}", self.content)?;
        writeln!(f, "标签: {:?}", self.tags)?;
        writeln!(f, "Token 数: {}", self.token_count)?;
        writeln!(f, "------------------------")
    }
}

impl FAQEntry {
    pub fn parse_from_markdown(markdown: &str) -> Vec<FAQEntry> {
        let mut entries = Vec::new();
        let mut current_category = "General".to_string();
        let mut pending_q: Option<String> = None;

        // 按行处理
        for line in markdown.lines() {
            let trimmed = line.trim();

            // 1.分类标题
            if trimmed.starts_with("## ") && !trimmed.starts_with("###") {
                let after_hash = trimmed.trim_start_matches("## ").trim();
                let category_clean = after_hash
                    .split(|c: char| c == '、' || c == '.')
                    .nth(1)
                    .unwrap_or(after_hash)
                    .trim()
                    .to_string();
                current_category = if category_clean.is_empty() {
                    after_hash.to_string()
                } else {
                    category_clean
                };
            }

            // 匹配 Q 行
            // 2. 匹配 Q 行
            if trimmed.starts_with("- Q") && trimmed.contains(": ") {
                let q_text = trimmed
                    .splitn(2, ':')
                    .nth(1)
                    .map(|s| s.trim().to_string())
                    .unwrap_or_default();

                pending_q = Some(q_text);
                continue;
            }

            // 3. 匹配 A 行（上一行是 Q）
            if let Some(q) = pending_q.take() {
                if trimmed.starts_with("A") && trimmed.contains(": ") {
                    let a_text = trimmed
                        .splitn(2, ':')
                        .nth(1)
                        .map(|s| s.trim().to_string())
                        .unwrap_or_default();

                    entries.push(FAQEntry {
                        category: current_category.clone(),
                        q,
                        a: a_text,
                        tags: vec![],
                    });
                } else {
                    pending_q = None;
                }
            }
        }

        entries
    }
}

#[cfg(test)]
mod tests { 
    use super::*;

    use std::fs;

    #[test]
    fn test_faq_chunking_from_file() {
        let faq_path = "/Users/xuenai/Code/rag-rs/docs/FAQ.md";
        let markdown = fs::read_to_string(faq_path)
            .expect("Failed to read FAQ.md");

        let entries = FAQEntry::parse_from_markdown(&markdown);
        println!("Parsed {} FAQ entries", entries.len());

        let chunker = FAQChunker::new(200, 30, "qwen-max".to_string()); // max_tokens=200, overlap=30, model="qwen-max"
        let chunks = chunker.chunk_by_qa(entries);

        println!("\nGenerated {} chunks:\n", chunks.len());
        for chunk in &chunks {
            println!("{}", chunk);
        }
    }

}