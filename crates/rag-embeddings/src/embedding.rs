use anyhow::Result;
use rag_indexing::tree_structrue::{LeafNode, NodeTree};

use crate::{client::{EmbeddingClient, qwen::QwenEmbeddingClient}, database::{VectorRecord, VectorStore, pgvector::PgVectorStore}};

// 叶子节点转为向量数据库中的记录 
pub fn leaf_to_vector_record(node_tree: &NodeTree, leaf: &LeafNode) -> VectorRecord {
    let hierarchy = &leaf.metadata.hierarchy;
    let parent_titles: Vec<String> = node_tree.get_ancestors(leaf.id)
        .into_iter()
        .filter_map(|node| node.title().map(|t|t.to_string()))
        .collect();

    VectorRecord {
        id: leaf.id.to_string(),
        embedding: leaf.embedding.clone().unwrap_or_default(),
        text: Some(leaf.text.clone()),
        metadata: serde_json::json!({
            "document_id": leaf.metadata.document_id,
            "node_id": leaf.id.to_string(),
            "chunk_index": leaf.metadata.hierarchy.last().and_then(|s| s.split('_').nth(1)).and_then(|s| s.parse::<i32>().ok()),
            "chunk_size": leaf.metadata.chunk_size,
            "file_name": leaf.metadata.file_name,
            "hierarchy": hierarchy,
            "parent_titles": parent_titles,
            "is_image": leaf.metadata.image_path.is_some(),
            "image_alt": leaf.metadata.image_alt,
            "image_path": leaf.metadata.image_path,
        }),
        createat: None,
        updateat: None,
    }
}

pub async fn save_node_tree(
    node_tree: &mut NodeTree,
    store: PgVectorStore,
    embedding_client: QwenEmbeddingClient,
) -> Result<()> {
    let mut texts = Vec::new();
    let mut leaf_ids = Vec::new();

    for leaf in node_tree.leaf_nodes() { 
        if leaf.embedding.is_none() {
            texts.push(leaf.text.clone());
            leaf_ids.push(leaf.id);
        }
    }

    if !texts.is_empty() {
        let embeddings = embedding_client.embed(texts).await?;
        for (i, embedding) in embeddings.into_iter().enumerate() {
            node_tree.set_leaf_embedding(leaf_ids[i], embedding)?;
        }
    }

    // match serde_json::to_string_pretty(node_tree) {
    //     Ok(json) => {
    //         println!("\n{} NODE TREE STRUCTURE (JSON) {}\n", "=".repeat(20), "=".repeat(20));
    //         println!("{}", json);
    //         println!("\n{}", "=".repeat(62));
    //     }
    //     Err(e) => eprintln!("序列化失败: {}", e),
    // }

    let records = node_tree
        .leaf_nodes()
        .filter(|leaf| leaf.embedding.is_some())
        .map(|leaf| leaf_to_vector_record(node_tree, leaf))
        .collect();

    store.upsert_vectors(records).await?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use anyhow::Result;
    use rag_indexing::tree_structrue::markdown_bulid::MarkdownParser;
    use sqlx::PgPool;
    use dotenv::dotenv;

    use crate::{client::qwen::QwenEmbeddingClient, database::pgvector::PgVectorStore, embedding::save_node_tree};

    const TEST: &str = r#"
# ChatGPT出现以来中美大模型发展报告
## 概述
自2022年11月30日OpenAI发布ChatGPT以来，全球人工智能领域发生了前所未有的变革。ChatGPT的发布标志着大语言模型时代的正式开启，激发了中美两国在AI领域的激烈竞争。这场始于技术突破的竞争，已经演变为涉及国家战略、产业生态、人才储备、基础设施等多个维度的全面博弈。
### 历史背景与意义

ChatGPT的出现并非偶然，而是人工智能发展到一定阶段的必然产物。在其发布之前，人工智能已经历了多次高潮与低谷：1950年代的符号主义、1980年代的专家系统、1990年代的机器学习兴起、2010年代的深度学习革命。每一次技术浪潮都推动着人工智能向更智能、更实用的方向发展，而ChatGPT所代表的大语言模型技术，则是这一历史进程的集大成者。

大语言模型的兴起有三大关键技术背景：首先，Transformer架构的提出为模型扩展提供了理论基础；其次，互联网上海量文本数据为模型训练提供了丰富素材；最后，GPU等并行计算设备的普及为大规模模型训练提供了硬件支撑。这些因素的结合，使得构建具有通用智能能力的大模型成为可能。
### 中美竞争的深层逻辑

中美两国在AI大模型领域的竞争，本质上是两种发展模式、两种技术路线、两种价值理念的较量。美国以硅谷为中心，代表了以企业为主导、市场驱动的创新模式，强调技术开源、生态开放、商业化优先。而中国则以北京、上海、深圳为核心，形成了以政府引导、产业协同、应用导向的发展模式，更加注重技术的产业化落地和社会效益。

这种竞争不仅体现在技术研发上，更深层次地反映了未来数字经济的主导权、人工智能标准制定的话语权、以及科技创新的主导地位谁能掌握。胜者将在未来几十年的全球科技竞争中占据有利位置，败者可能面临技术依赖和产业边缘化的风险。
    "#;

    #[tokio::test]
    async fn test() -> Result<()> {
        dotenv().ok();
        let api_key = std::env::var("DASHSCOPE_API_KEY")
            .expect("请设置环境变量 DASHSCOPE_API_KEY 或在 .env 文件中配置");
        let embedding_client = QwenEmbeddingClient::for_text(api_key, "text-embedding-v1".to_string());


        let parser = MarkdownParser::new("doc-001".to_string(),Some("test.md".to_string()));
        let mut tree = parser.parse(TEST)?;

        let pool = PgPool::connect("postgres:///rag_db").await?;
        let store = PgVectorStore::new(pool, "vectors", 1536).await?;
        save_node_tree(&mut tree, store, embedding_client).await?;
        Ok(())
    }
    
}