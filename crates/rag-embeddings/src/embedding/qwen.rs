use crate::embedding::{EmbeddingClient, EmbeddingError, EmbeddingResult};
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
struct OpenAIEmbeddingResponse {
    data: Vec<OpenAIEmbeddingItem>,
    _model: String,
    _usage: OpenAIUsage,
}

#[derive(Deserialize, Debug)]
struct OpenAIEmbeddingItem {
    embedding: Vec<f32>,
    index: usize,
}

#[derive(Deserialize, Debug)]
struct OpenAIUsage {
    _prompt_tokens: usize,
    _total_tokens: usize,
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
            input: texts,
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
            .map_err(|e| EmbeddingError::Network(e.to_string()))?;

        let status = resp.status();
        let resp_text = resp.text().await.map_err(|e| EmbeddingError::Network(e.to_string()))?;

        if !status.is_success() {
            if let Ok(err_resp) = serde_json::from_str::<ErrorResponse>(&resp_text) {
                let msg = err_resp.error.message.unwrap_or("Unknown error".to_string());
                let code = err_resp.error.code.unwrap_or_default();
                return Err(EmbeddingError::Api(format!("[{}] {}", code, msg)));
            } else {
                return Err(EmbeddingError::Api(format!("HTTP {}: {}", status, resp_text.trim())));
            }
        }

        let openai_resp: OpenAIEmbeddingResponse = serde_json::from_str(&resp_text)
            .map_err(|e| EmbeddingError::InvalidResponse(e.to_string()))?;

        let mut embeddings: Vec<_> = openai_resp.data.into_iter().collect();
        embeddings.sort_by_key(|item| item.index);
        let vectors: Vec<Vec<f32>> = embeddings.into_iter().map(|item| item.embedding).collect();

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