use once_cell::sync::Lazy;
use regex::Regex;
use std::collections::HashMap;
use std::fmt;
use tiktoken_rs::CoreBPE;


#[derive(Debug, Clone)]
pub struct TextChunk {
    pub content: String,
    pub page_number: usize,
    pub chunk_index: usize,
    pub char_range: (usize, usize),
    pub metadata: HashMap<String, String>,
}

#[derive(Clone)]
pub struct RecursiveChunker {
    max_tokens: usize,
    model: String,
    bpe: CoreBPE,
}

impl fmt::Debug for RecursiveChunker {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let mut debug_struct = f.debug_struct("RecursiveChunker");
        debug_struct.field("max_tokens", &self.max_tokens);
        debug_struct.field("model", &self.model);
        debug_struct.field("bpe", &"CoreBPE");
        debug_struct.finish()
    }
}

impl RecursiveChunker {
    /// 创建分块器
    pub fn new(max_tokens: usize, model: &str) -> Self {
        let key = Self::normalize_model(model);
        let bpe = tiktoken_rs::get_bpe_from_model(&key)
            .expect(&format!("无法为模型 {} 创建 tokenizer（标准化后: {}）", model, key));

        Self {
            max_tokens,
            model: model.to_string(),
            bpe,
        }
    }

    /// 递归分块主函数
    pub fn chunk(&self, text_with_pages: Vec<(usize, String)>) -> Vec<TextChunk> {
        let mut chunks = Vec::new();
        let mut global_offset = 0;
        let mut chunk_index = 0;

        for (page, page_text) in text_with_pages {
            let paragraphs = self.split_paragraphs(&page_text);

            for para in paragraphs {
                let para_len = para.len();
                if self.token_count(&para) <= self.max_tokens {
                    // 小段落直接成块
                    chunks.push(self.make_chunk(
                        &para,
                        page,
                        global_offset,
                        chunk_index,
                    ));
                    chunk_index += 1;
                    global_offset += para_len + 1;
                } else {
                    // 递归切分
                    let subchunks = self.recursive_split(&para, page, global_offset, &mut chunk_index);
                    chunks.extend(subchunks);
                    global_offset += para_len + 1;
                }
            }
        }

        chunks
    }

    /// 递归切分大段落
    fn recursive_split(
        &self,
        text: &str,
        page: usize,
        start_offset: usize,
        chunk_index: &mut usize,
    ) -> Vec<TextChunk> {
        let mut chunks = Vec::new();
        let mut buffer = String::new();
        let mut current_offset = start_offset;

        // 按句子切分
        let sentences = self.split_sentences(text);

        for sentence in sentences {
            let sent = sentence.trim();
            if sent.is_empty() { continue; }

            let new_buffer = if buffer.is_empty() {
                sent.to_string()
            } else {
                format!("{} {}", buffer, sent)
            };

            // 检查 token 数
            if self.token_count(&new_buffer) <= self.max_tokens {
                buffer = new_buffer;
            } else {
                // 提交当前 buffer
                if !buffer.is_empty() {
                    chunks.push(self.make_chunk(&buffer, page, current_offset, *chunk_index));
                    *chunk_index += 1;
                    current_offset += buffer.len() + 1;
                }
                // 新句子单独成块（如果太长，再递归）
                if self.token_count(sent) <= self.max_tokens {
                    chunks.push(self.make_chunk(sent, page, current_offset, *chunk_index));
                    *chunk_index += 1;
                    current_offset += sent.len() + 1;
                    buffer.clear();
                } else {
                    // 极端长句：按字符硬切
                    let hard_chunks = self.hard_split(sent, page, current_offset, chunk_index);
                    chunks.extend(hard_chunks.clone());
                    let total_len: usize = hard_chunks.iter().map(|c| c.content.len() + 1).sum();
                    current_offset += total_len;
                    *chunk_index += hard_chunks.len();
                    buffer.clear();
                }
            }
        }

        // 最后一块
        if !buffer.is_empty() {
            chunks.push(self.make_chunk(&buffer, page, current_offset, *chunk_index));
            *chunk_index += 1;
        }

        chunks
    }

    /// 按段落切分（空行分隔）
    fn split_paragraphs(&self, text: &str) -> Vec<String> {
        text.split("\n\n")
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect()
    }

    /// 按句子切分（中英文）
    fn split_sentences<'a>(&self, text: &'a str) -> Vec<&'a str> {
        static CN_SENT: Lazy<Regex> = 
            Lazy::new(|| Regex::new(r"[。！？\n]+").unwrap());
        static EN_SENT: Lazy<Regex> = 
            Lazy::new(|| Regex::new(r"[.!?\n]+").unwrap());

        let mut sentences = Vec::new();
        let mut start = 0;

        // 优先中文标点
        for mat in CN_SENT.find_iter(text) {
            if mat.start() > start {
                sentences.push(text[start..mat.start()].trim());
            }
            start = mat.end();
        }
        if start < text.len() {
            sentences.push(text[start..].trim());
        }

        // 如果没中文标点，用英文
        if sentences.len() <= 1 {
            sentences.clear();
            start = 0;
            for mat in EN_SENT.find_iter(text) {
                if mat.start() > start {
                    sentences.push(text[start..mat.start()].trim());
                }
                start = mat.end();
            }
            if start < text.len() {
                sentences.push(text[start..].trim());
            }
        }

        // 过滤空串
        sentences
            .into_iter()
            .filter(|s| !s.is_empty())
            .collect()
    }

    /// 极端长句：按字符硬切
    fn hard_split(
        &self,
        text: &str,
        page: usize,
        start_offset: usize,
        chunk_index: &mut usize,
    ) -> Vec<TextChunk> {
        let mut chunks = Vec::new();
        let chars: Vec<char> = text.chars().collect();
        let mut i = 0;
        let mut current_offset = start_offset;

        while i < chars.len() {
            let mut end = i + 500; // 每次最多 500 字符
            if end > chars.len() { end = chars.len(); }

            // 尽量在空格或标点处断开
            while end > i && !Self::is_good_break(chars[end - 1]) {
                end -= 1;
            }
            if end == i { end = i + 300; } // 强制断开

            let slice = chars[i..end].iter().collect::<String>();
            chunks.push(self.make_chunk(&slice, page, current_offset, *chunk_index));
            *chunk_index += 1;
            current_offset += slice.len() + 1;
            i = end;
        }

        chunks
    }

    fn is_good_break(c: char) -> bool {
        c.is_whitespace() || matches!(c, '，' | ',' | '；' | ';' | '：' | ':' | ' ' | '\n')
    }

    /// 创建 chunk
    fn make_chunk(&self, content: &str, page: usize, offset: usize, index: usize) -> TextChunk {
        TextChunk {
            content: content.to_string(),
            page_number: page,
            chunk_index: index,
            char_range: (offset, offset + content.len()),
            metadata: HashMap::from([
                ("model".to_string(), self.model.clone()),
                ("token_count".to_string(), self.token_count(content).to_string()),
            ]),
        }
    }

    /// 计算 token 数（使用模型原生的 tokenizer）
    fn token_count(&self, text: &str) -> usize {
        self.bpe.encode_with_special_tokens(text).len()
    }

    /// 标准化模型名（支持别名）
    fn normalize_model(model: &str) -> String {
        match model.trim().to_lowercase().as_str() {
            "gpt-4o" | "gpt-4" | "gpt-4-turbo" | "gpt-4o-mini" => "gpt-4o".to_string(),
            "gpt-3.5" | "gpt-3.5-turbo" | "chatgpt" => "gpt-3.5-turbo".to_string(),
            "text-embedding-3-small" | "embedding-small" => "text-embedding-3-small".to_string(),
            "text-embedding-3-large" | "embedding-large" => "text-embedding-3-large".to_string(),
            "text-embedding-ada-002" | "ada" => "text-embedding-ada-002".to_string(),
            // Qwen 系列（使用 cl100k_base 编码，与 GPT-4 兼容）
            "qwen" | "qwen-max" | "qwen-plus" | "qwen-turbo" | "qwen-7b" | "qwen-14b" | "qwen-72b" => "gpt-4o".to_string(),
            _ => model.to_string(),
        }
    }
}

#[cfg(test)]

mod tests {
    use super::*;
    use std::fs;
    use anyhow::Result;
    use std::path::Path;
    #[test]
    pub fn test_count_tokens() -> Result<()> {

        let path = Path::new("/Users/xuenai/Code/rag-rs/docs/google.txt");
        let text = fs::read_to_string(path).expect("无法读取");

        let chunker = RecursiveChunker::new(512, "gpt-3.5-turbo");
        let chunks = chunker.chunk(vec![(1, text)]);

        println!("\n=== 分块结果（共 {} 块）===", chunks.len());
        for (i, chunk) in chunks.iter().enumerate() {
            println!(
                "[{}] Page {} | {} tokens | {}..{}",
                i,
                chunk.page_number,
                chunk.metadata["token_count"],
                chunk.char_range.0,
                chunk.char_range.1
            );
            println!("   > {}\n", chunk.content);
        }
        Ok(())
    }
}