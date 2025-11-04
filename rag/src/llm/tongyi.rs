use anyhow::{anyhow, Result};
use async_openai::types::{ChatCompletionRequestMessage, CreateChatCompletionRequestArgs};
use async_trait::async_trait;
use dotenv::dotenv;

use crate::llm::LlmClient;

pub struct TongyiClient {
    pub api_key: String,
    pub base_url: String,
    pub model: String,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub client: reqwest::Client,
}


impl TongyiClient {
    
    pub fn new() -> Self {
        dotenv().ok();
        let api_key = std::env::var("DASHSCOPE_API_KEY")
            .expect("请设置环境变量 DASHSCOPE_API_KEY");
        Self {
            api_key,
            base_url: "https://dashscope.aliyuncs.com/compatible-mode/v1".to_string(),
            model: "qwen-max".to_string(),
            max_tokens: Some(10000),
            temperature: Some(0.7),
            client: reqwest::Client::new(),
        }
    }

    pub fn with_model(mut self, model: String) -> Self {
        self.model = model;
        self
    }

    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }
}

#[async_trait]
impl LlmClient for TongyiClient {
    async fn chat(&self, messages: Vec<ChatCompletionRequestMessage>) -> Result<String> {
        // 构建请求参数
        let request = CreateChatCompletionRequestArgs::default()
            .model(self.model.clone())
            .messages(messages)
            .max_tokens(self.max_tokens.unwrap_or(10000))
            .temperature(self.temperature.unwrap_or(0.7))
            .build()?;

        // 发送请求
        let url = format!("{}/chat/completions", self.base_url);
        let response = self.client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&request)
            .send()
            .await?;

        // 检查响应状态
        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await.unwrap_or_default();
            return Err(anyhow!("API请求失败: {} - {}", status, error_text));
        }

        // 解析响应
        let response_text = response.text().await?;
        let response_json: serde_json::Value = serde_json::from_str(&response_text)?;

        // 提取返回的消息内容
        if let Some(choices) = response_json["choices"].as_array() {
            if let Some(first_choice) = choices.first() {
                if let Some(content) = first_choice["message"]["content"].as_str() {
                    return Ok(content.to_string());
                }
            }
        }

        Err(anyhow!("无法从响应中提取消息内容: {}", response_text))
    }

    async fn generate(&self, messages: Vec<ChatCompletionRequestMessage>) -> Result<String> {
        // generate方法可以复用chat方法
        self.chat(messages).await
    }
}