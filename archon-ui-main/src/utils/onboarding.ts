export interface NormalizedCredential {
  key: string;
  value?: string;
  is_encrypted?: boolean;
  category: string;
}

export interface ProviderInfo {
  provider?: string;
}

/**
 * Determines if LM (Language Model) is configured based on credentials
 * 
 * Logic:
 * - provider := value of 'LLM_PROVIDER' from ragCreds (if present)
 * - if provider === 'openai': return true iff apiKeyCreds includes key 'OPENAI_API_KEY' with (value || is_encrypted)
 * - if provider && provider !== 'openai': return true (other providers like Ollama don't need API keys)
 * - if no provider: return true iff apiKeyCreds includes 'OPENAI_API_KEY' (value || is_encrypted)
 */
export function isLmConfigured(
  ragCreds: NormalizedCredential[],
  apiKeyCreds: NormalizedCredential[]
): boolean {
  // Find the LLM_PROVIDER setting from RAG credentials
  const providerCred = ragCreds.find(c => c.key === 'LLM_PROVIDER');
  const provider = providerCred?.value;

  // Check if OpenAI API key exists (either as value or encrypted)
  const hasOpenAIKey = apiKeyCreds.some(
    (c) =>
      c.key.toUpperCase() === 'OPENAI_API_KEY' && (c.value || c.is_encrypted)
  );

  if (provider === 'openai') {
    // OpenAI provider requires an API key
    return hasOpenAIKey;
  } else if (provider && provider !== 'openai') {
    // Other providers (e.g., 'ollama', 'google') are considered configured
    return true;
  } else {
    // No provider specified, default behavior: check for OpenAI key
    return hasOpenAIKey;
  }
}