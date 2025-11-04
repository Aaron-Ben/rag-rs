use tiktoken_rs::{get_bpe_from_model, CoreBPE};
use once_cell::sync::Lazy;
use std::collections::HashMap;

/// 全局缓存：模型名 → BPE 编码器（线程安全、高性能）
static BPE_CACHE: Lazy<std::sync::Mutex<HashMap<String, CoreBPE>>> = Lazy::new(|| {
    std::sync::Mutex::new(HashMap::new())
});

/// 计算文本的 token 数量
/// 
/// # 参数
/// - `text`: 输入文本
/// - `model`: 模型名，如 "gpt-4o", "gpt-3.5-turbo", "text-embedding-3-small"
/// 
/// # 返回
/// `usize` token 数量
pub fn count_tokens(text: &str, model: &str) -> usize {
    // 标准化模型名
    let model_key = normalize_model_name(model);

    // 获取或创建 BPE 编码器
    let bpe = {
        let mut cache = BPE_CACHE.lock().unwrap();
        cache.entry(model_key.clone())
            .or_insert_with(|| get_bpe_from_model(&model_key).expect("Unsupported model"))
            .clone()
    };

    // 编码并计数
    bpe.encode_with_special_tokens(text).len()
}

/// 标准化模型名（支持别名）
fn normalize_model_name(model: &str) -> String {
    match model.trim().to_lowercase().as_str() {
        // GPT-4 系列
        "gpt-4" | "gpt-4-turbo" | "gpt-4o" | "gpt-4o-mini" => "gpt-4o".to_string(),
        // GPT-3.5 系列
        "gpt-3.5" | "gpt-3.5-turbo" | "chatgpt" => "gpt-3.5-turbo".to_string(),
        // 嵌入模型
        "text-embedding-3-small" | "embedding-small" => "text-embedding-3-small".to_string(),
        "text-embedding-3-large" | "embedding-large" => "text-embedding-3-large".to_string(),
        "text-embedding-ada-002" | "ada" => "text-embedding-ada-002".to_string(),
        // 默认
        _ => model.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;  
    #[test]
    pub fn test_count_tokens() {
        let text = "Rust 是一门系统编程语言，专注于安全与性能。\n它由 Mozilla 开发。";

        let models = [
            "gpt-4o",
            "gpt-3.5-turbo",
            "text-embedding-3-small",
        ];

        for model in models {
            let tokens = count_tokens(text, model);
            println!("[{}] tokens: {}", model, tokens);
        }
    }
}