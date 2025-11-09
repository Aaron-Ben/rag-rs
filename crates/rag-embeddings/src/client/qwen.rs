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
    /// 是否启用归一化
    normalize: bool,
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
            normalize: true, // 启用归一化
        }
    }

    pub fn for_text(api_key: String, model: String) -> Self {
        Self::new(api_key, model, Some("retrieval.document".to_string()))
    }
    
    /// L2 归一化单个 embedding 向量
    /// 将向量投影到单位球面上，确保 ||v|| = 1.0
    fn normalize_embedding(&self, embedding: &mut Vec<f32>) -> Result<(), EmbeddingError> {
        if !self.normalize {
            return Ok(());
        }

        if embedding.is_empty() {
            return Err(EmbeddingError::InvalidVector("Empty embedding vector".to_string()));
        }

        // 计算 L2 范数：sqrt(∑(x_i²))
        let norm: f64 = embedding.iter()
            .map(|&x| (x as f64).powi(2))
            .sum::<f64>()
            .sqrt();

        let norm_f32 = norm as f32;
        
        if norm_f32.abs() < 1e-8 {
            return Err(EmbeddingError::InvalidVector("Zero vector cannot be normalized".to_string()));
        }

        // 归一化：v_i = v_i / ||v||
        for value in embedding.iter_mut() {
            *value /= norm_f32;
        }

        Ok(())
    }

    /// 批量归一化多个 embedding 向量
    fn normalize_vectors(&self, embeddings: &mut Vec<Vec<f32>>) -> Result<(), EmbeddingError> {
        for embedding in embeddings.iter_mut() {
            self.normalize_embedding(embedding)?;
        }
        Ok(())
    }

    /// 验证向量的归一化状态
    /// 检查 L2 范数是否接近 1.0（容差 1e-6）
    pub fn is_normalized(&self, embedding: &Vec<f32>) -> bool {
        if embedding.is_empty() {
            return false;
        }

        let norm: f64 = embedding.iter()
            .map(|&x| (x as f64).powi(2))
            .sum::<f64>()
            .sqrt();

        let tolerance = 1e-6;
        (norm - 1.0).abs() < tolerance
    }

    /// 获取客户端配置信息
    pub fn info(&self) -> String {
        format!(
            "QwenEmbeddingClient: model={}, dimension={}, normalize={}",
            self.model, self.dimension, self.normalize
        )
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
        let mut vectors: Vec<Vec<f32>> = if let Some(embeddings) = value.get("data").and_then(|d| d.as_array()) {
            // OpenAI 兼容格式
            let mut embeds: Vec<(usize, Vec<f32>)> = Vec::new();
            for item in embeddings {
                if let (Some(index), Some(embedding_array)) = (
                    item.get("index").and_then(|i| i.as_u64()),
                    item.get("embedding").and_then(|e| e.as_array()),
                ) {
                    let mut embedding: Vec<f32> = embedding_array
                        .iter()
                        .filter_map(|v| v.as_f64().map(|f| f as f32))
                        .collect();
                    
                    // 立即归一化单个向量
                    self.normalize_embedding(&mut embedding)?;
                    
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
            let mut embeds: Vec<Vec<f32>> = Vec::new();
            for item in embeddings {
                if let Some(embedding_array) = item.get("embedding").and_then(|e| e.as_array()) {
                    let mut embedding: Vec<f32> = embedding_array
                        .iter()
                        .filter_map(|v| v.as_f64().map(|f| f as f32))
                        .collect();
                    
                    // 立即归一化单个向量
                    self.normalize_embedding(&mut embedding)?;
                    
                    embeds.push(embedding);
                }
            }
            embeds
        } else {
            return Err(EmbeddingError::InvalidResponse(
                "无法从响应中提取 embedding 数据".to_string()
            ));
        };

        // 确保所有向量都已归一化（冗余检查）
        self.normalize_vectors(&mut vectors)?;

        // 验证归一化结果
        for (i, embedding) in vectors.iter().enumerate() {
            if !self.is_normalized(embedding) {
                println!("警告: 向量 {} 归一化失败，L2 范数: {:.6}", 
                    i, embedding.iter().map(|&x| x as f64 * x as f64).sum::<f64>().sqrt());
            }
        }

        println!("✅ 已生成 {} 个归一化向量，每个维度: {}", vectors.len(), self.dimension);
        
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
    use anyhow::Result;

    #[tokio::test]
    async fn test_embed() -> Result<()> {
        dotenv().ok();
        let api_key = std::env::var("DASHSCOPE_API_KEY")
            .expect("请设置环境变量 DASHSCOPE_API_KEY 或在 .env 文件中配置");
        
        let client = QwenEmbeddingClient::for_text(api_key, "text-embedding-v1".to_string());
        let texts = vec!["Hello, world!".to_string(), "Rust is awesome!".to_string()];
        
        println!("客户端信息: {}", client.info());
        
        let embeddings = client.embed(texts.clone()).await?;
        
        println!("生成了 {} 个 embedding，向量维度: {}", embeddings.len(), embeddings[0].len());
        
        // 验证每个向量的维度
        for (i, embedding) in embeddings.iter().enumerate() {
            assert_eq!(embedding.len(), client.dimension(), "向量 {} 维度不匹配", i);
            
            // 验证归一化
            let is_norm = client.is_normalized(embedding);
            let norm = embedding.iter().map(|&x| x as f64 * x as f64).sum::<f64>().sqrt();
            
            println!("向量 {}: 维度={}, 归一化={}, L2范数={:.8}", 
                i, embedding.len(), is_norm, norm);
            
            assert!(is_norm, "向量 {} 未正确归一化", i);
            assert!((norm - 1.0).abs() < 1e-6, "向量 {} L2 范数没有在正确的范围", i);
        }
        
        println!("✅ 所有测试通过！");
        Ok(())
    }

    #[tokio::test]
    async fn test_empty_input() {
        dotenv().ok();
        let api_key = std::env::var("DASHSCOPE_API_KEY")
            .expect("请设置环境变量 DASHSCOPE_API_KEY 或在 .env 文件中配置");
        let client = QwenEmbeddingClient::for_text(api_key, "text-embedding-v1".to_string());
        
        let result = client.embed(vec![]).await;
        assert!(result.is_err());
        if let Err(EmbeddingError::Api(msg)) = result {
            assert_eq!(msg, "Input texts cannot be empty");
        } else {
            panic!("Expected Api error for empty input");
        }
    }

    #[tokio::test]
    async fn test_zero_vector_normalization() {
        dotenv().ok();
        let api_key = std::env::var("DASHSCOPE_API_KEY")
            .expect("请设置环境变量 DASHSCOPE_API_KEY 或在 .env 文件中配置");
        let client = QwenEmbeddingClient::for_text(api_key, "text-embedding-v1".to_string());
        
        let mut zero_vector = vec![0.0f32; 1536];
        let result = client.normalize_embedding(&mut zero_vector);
        
        assert!(result.is_err());
        if let Err(EmbeddingError::InvalidVector(msg)) = result {
            assert!(msg.contains("Zero vector"));
        }
    }
}