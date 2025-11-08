use crate::client::{EmbeddingClient, EmbeddingError, EmbeddingResult};
use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};

#[derive(Serialize)]
struct QwenRequest {
    model: String,
    input: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    task: Option<String>,
}

#[derive(Deserialize, Debug)]
struct DashScopeError {
    code: Option<String>,
    message: Option<String>,
}

#[derive(Deserialize, Debug)]
struct ErrorResponse {
    error: DashScopeError,
}

pub struct QwenEmbeddingClient {
    api_key: String,
    model: String,
    task: Option<String>,
    client: Client,
    dimension: usize,
}

impl QwenEmbeddingClient {
    pub fn new(api_key: String, model: String, task: Option<String>) -> Self {
        let dimension = match model.as_str() {
            "text-embedding-v1" => 1536,
            "text-embedding-v2" => 1536,
            "text-embedding-v3" => 2560,
            _ => 1536,
        };

        Self {
            api_key,
            model,
            task,
            client: Client::new(),
            dimension,
        }
    }

    pub fn for_text(api_key: String, model: String) -> Self {
        Self::new(api_key, model, Some("retrieval.document".to_string()))
    }
}

#[async_trait]
impl EmbeddingClient for QwenEmbeddingClient {
    async fn embed(&self, texts: Vec<String>) -> EmbeddingResult<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Err(EmbeddingError::Api("Input texts cannot be empty".to_string()));
        }

        let request = QwenRequest {
            model: self.model.clone(),
            input: texts.clone(),
            task: self.task.clone(),
        };

        const QWEN_EMBEDDING_API: &str = "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings";

        let resp = self.client
            .post(QWEN_EMBEDDING_API)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await
            .map_err(|e| {
                println!("网络请求错误: {}", e);
                EmbeddingError::Network(e.to_string())
            })?;

        let status = resp.status();
        let resp_text = resp.text().await.map_err(|e| {
            println!("读取响应文本错误: {}", e);
            EmbeddingError::Network(e.to_string())
        })?;


        if !status.is_success() {
            println!("API 返回错误状态");
            if let Ok(err_resp) = serde_json::from_str::<ErrorResponse>(&resp_text) {
                let msg = err_resp.error.message.unwrap_or("Unknown error".to_string());
                let code = err_resp.error.code.unwrap_or_default();
                return Err(EmbeddingError::Api(format!("[{}] {}", code, msg)));
            } else {
                return Err(EmbeddingError::Api(format!("HTTP {}: {}", status, resp_text.trim())));
            }
        }

        // 使用 Value 来动态解析
        let value: serde_json::Value = serde_json::from_str(&resp_text)
            .map_err(|e| {
                println!("JSON 解析错误: {}", e);
                EmbeddingError::InvalidResponse(e.to_string())
            })?;

        // println!("解析后的 JSON: {:#}", value);

        // 根据实际响应结构提取 embeddings
        let vectors:Vec<Vec<f32>> = if let Some(embeddings) = value.get("data").and_then(|d| d.as_array()) {
            // OpenAI 兼容格式
            let mut embeds: Vec<(usize, Vec<f32>)> = Vec::new();
            for item in embeddings {
                if let (Some(index), Some(embedding_array)) = (
                    item.get("index").and_then(|i| i.as_u64()),
                    item.get("embedding").and_then(|e| e.as_array()),
                ) {
                    let embedding: Vec<f32> = embedding_array
                        .iter()
                        .filter_map(|v| v.as_f64().map(|f| f as f32))
                        .collect();
                    embeds.push((index as usize, embedding));
                }
            }
            embeds.sort_by_key(|(index, _)| *index);
            embeds.into_iter().map(|(_, embedding)| embedding).collect()
        } else if let Some(embeddings) = value.get("output")
            .and_then(|o| o.get("embeddings"))
            .and_then(|e| e.as_array()) 
        {
            // 达摩院原生格式
            let mut embeds: Vec<(usize, Vec<f32>)> = Vec::new();
            for (i, item) in embeddings.iter().enumerate() {
                if let Some(embedding_array) = item.get("embedding").and_then(|e| e.as_array()) {
                    let embedding: Vec<f32> = embedding_array
                        .iter()
                        .filter_map(|v| v.as_f64().map(|f| f as f32))
                        .collect();
                    embeds.push((i, embedding));
                }
            }
            embeds.into_iter().map(|(_, embedding)| embedding).collect()
        } else {
            return Err(EmbeddingError::InvalidResponse(
                "无法从响应中提取 embedding 数据".to_string()
            ));
        };

        Ok(vectors)
    }

    fn dimension(&self) -> usize {
        self.dimension
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio;
    use dotenv::dotenv;

    #[tokio::test]
    async fn test_embed() {
        dotenv().ok();
        let api_key = std::env::var("DASHSCOPE_API_KEY")
            .expect("请设置环境变量 DASHSCOPE_API_KEY 或在 .env 文件中配置");
        
        let client = QwenEmbeddingClient::for_text(api_key, "text-embedding-v1".to_string());
        let texts = vec!["Hello, world!".to_string(), "Rust is awesome!".to_string()];
        let embeddings = client.embed(texts).await;
        println!("{:?}", embeddings);
    }
    
}