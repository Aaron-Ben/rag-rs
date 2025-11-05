use serde::{Deserialize, Serialize};
use std::fmt;
use crate::tiktoken::count_tokens;

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
        let sentences: Vec<&str> = text
            .split(&['。', '！', '？', '.', '!', '?', '；', ';'])
            .filter(|s| !s.trim().is_empty())
            .collect();

        let mut current_chunk_idx = 1;
        let mut current_sentences = Vec::new();
        let mut current_token_count = 0;

        for (sent_idx, sentence) in sentences.iter().enumerate() {
            let sentence_with_punc = format!("{}{}", sentence.trim(), "。"); // 补回标点
            let sentence_tokens = self.count_tokens(&sentence_with_punc);

            // 预判：添加当前句子后是否超 max_tokens
            let new_token_count = current_token_count + sentence_tokens;
            if new_token_count > self.max_tokens && !current_sentences.is_empty() {
                // 生成当前 chunk
                let chunk_content = current_sentences.join("");
                chunks.push(FAQChunk {
                    chunk_id: format!("{}-chunk-{}", faq_id, current_chunk_idx),
                    faq_id: faq_id.to_string(),
                    category: entry.category.clone(),
                    title: entry.q.trim().to_string(),
                    content: chunk_content,
                    tags: entry.tags.clone(),
                    token_count: current_token_count,
                });

                // 重置当前句子（保留重叠部分，避免语义断裂）
                current_sentences.clear();
                if self.overlap > 0 {
                    // 从当前句子往前取 overlap 个句子作为重叠
                    let start_idx = if sent_idx >= self.overlap {
                        sent_idx - self.overlap
                    } else {
                        0
                    };
                    for s in &sentences[start_idx..=sent_idx] {
                        current_sentences.push(format!("{}{}", s.trim(), "。"));
                    }
                    // 重新计算重叠后的 token 数
                    current_token_count = self.count_tokens(
                        &current_sentences.join("")
                    );
                } else {
                    current_sentences.push(sentence_with_punc.clone());
                    current_token_count = sentence_tokens;
                }
                current_chunk_idx += 1;
            } else {
                current_sentences.push(sentence_with_punc);
                current_token_count = new_token_count;
            }
        }

        // 添加最后一个 chunk
        if !current_sentences.is_empty() {
            let chunk_content = current_sentences.join("");
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