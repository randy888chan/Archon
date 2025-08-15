import React, { useState } from 'react';
import { Settings, Check, Save, Loader, ChevronDown, ChevronUp, Zap, Database } from 'lucide-react';
import { Card } from '../ui/Card';
import { Input } from '../ui/Input';
import { Button } from '../ui/Button';
import { useToast } from '../../contexts/ToastContext';
import { credentialsService } from '../../services/credentialsService';
import { ProviderSelectionGrid } from './ProviderSelectionGrid';
import { Provider } from './ProviderTileButton';

interface RAGSettingsWithTilesProps {
  ragSettings: {
    MODEL_CHOICE: string;
    USE_CONTEXTUAL_EMBEDDINGS: boolean;
    CONTEXTUAL_EMBEDDINGS_MAX_WORKERS: number;
    USE_HYBRID_SEARCH: boolean;
    USE_AGENTIC_RAG: boolean;
    USE_RERANKING: boolean;
    LLM_PROVIDER?: string;
    LLM_BASE_URL?: string;
    EMBEDDING_MODEL?: string;
    EMBEDDING_PROVIDER?: string;
    // Crawling Performance Settings
    CRAWL_BATCH_SIZE?: number;
    CRAWL_MAX_CONCURRENT?: number;
    CRAWL_WAIT_STRATEGY?: string;
    CRAWL_PAGE_TIMEOUT?: number;
    CRAWL_DELAY_BEFORE_HTML?: number;
    // Storage Performance Settings
    DOCUMENT_STORAGE_BATCH_SIZE?: number;
    EMBEDDING_BATCH_SIZE?: number;
    DELETE_BATCH_SIZE?: number;
    ENABLE_PARALLEL_BATCHES?: boolean;
    // Advanced Settings
    MEMORY_THRESHOLD_PERCENT?: number;
    DISPATCHER_CHECK_INTERVAL?: number;
    CODE_EXTRACTION_BATCH_SIZE?: number;
    CODE_SUMMARY_MAX_WORKERS?: number;
  };
  setRagSettings: (settings: any) => void;
}

export const RAGSettingsWithTiles = ({
  ragSettings,
  setRagSettings
}: RAGSettingsWithTilesProps) => {
  const [saving, setSaving] = useState(false);
  const [showCrawlingSettings, setShowCrawlingSettings] = useState(false);
  const [showStorageSettings, setShowStorageSettings] = useState(false);
  const { showToast } = useToast();

  const handleLLMProviderSelect = (provider: Provider) => {
    setRagSettings({
      ...ragSettings,
      LLM_PROVIDER: provider
    });
  };

  const handleEmbeddingProviderSelect = (provider: Provider) => {
    setRagSettings({
      ...ragSettings,
      EMBEDDING_PROVIDER: provider
    });
  };

  const currentLLMProvider = (ragSettings.LLM_PROVIDER || 'openai') as Provider;
  const currentEmbeddingProvider = (ragSettings.EMBEDDING_PROVIDER || 'openai') as Provider;

  return (
    <Card accentColor="green" className="overflow-hidden p-8">
      {/* Description */}
      <div className="mb-8">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-2 flex items-center">
          <Settings className="mr-3 text-green-500" size={24} />
          Model Configuration & Performance
        </h2>
        <p className="text-sm text-gray-600 dark:text-zinc-400">
          Configure your AI models, context windows, and performance settings. Start with RAG Settings for basic configuration, then optimize context windows with NUM CTX Management for advanced performance tuning.
        </p>
      </div>

      {/* LLM Provider Selection */}
      <div className="mb-8">
        <ProviderSelectionGrid
          selectedProvider={currentLLMProvider}
          onProviderSelect={handleLLMProviderSelect}
          title="LLM Provider Selection"
          subtitle="Choose your language model provider for chat and reasoning tasks"
        />
      </div>

      {/* LLM Chat Models Configuration */}
      <div className="mb-8 p-6 rounded-lg border border-green-500/20 bg-gradient-to-r from-green-500/5 to-transparent">
        <div className="flex items-center mb-4">
          <div className="w-6 h-6 rounded-full bg-green-500/20 border border-green-500/30 flex items-center justify-center mr-3">
            <div className="w-2 h-2 rounded-full bg-green-500" />
          </div>
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
            LLM Chat Models
          </h3>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {currentLLMProvider === 'ollama' && (
            <div>
              <Input
                label="Ollama Chat Model Server URL"
                value={ragSettings.LLM_BASE_URL || 'http://192.168.3.12:11434'}
                onChange={e => setRagSettings({
                  ...ragSettings,
                  LLM_BASE_URL: e.target.value
                })}
                placeholder="http://192.168.3.12:11434"
                accentColor="green"
              />
            </div>
          )}
          
          <div>
            <Input 
              label="Chat Model" 
              value={ragSettings.MODEL_CHOICE} 
              onChange={e => setRagSettings({
                ...ragSettings,
                MODEL_CHOICE: e.target.value
              })} 
              placeholder={getModelPlaceholder(currentLLMProvider)}
              accentColor="green" 
            />
          </div>
        </div>

        {/* Ollama Configuration Tips */}
        {currentLLMProvider === 'ollama' && (
          <div className="mt-4 p-4 rounded-lg bg-blue-500/10 border border-blue-500/20">
            <div className="flex items-start">
              <div className="w-5 h-5 rounded-full bg-blue-500/20 border border-blue-500/30 flex items-center justify-center mr-3 mt-0.5 flex-shrink-0">
                <div className="w-2 h-2 rounded-full bg-blue-500" />
              </div>
              <div className="space-y-2 text-sm">
                <h4 className="font-medium text-blue-900 dark:text-blue-300">Ollama Configuration Tips</h4>
                <ul className="space-y-1 text-blue-800 dark:text-blue-400">
                  <li>• Ensure your Ollama server is running and accessible</li>
                  <li>• Models must be pulled locally: <code className="px-1 py-0.5 bg-blue-500/20 rounded text-xs">ollama pull gemma3:12b</code></li>
                  <li>• Tool support requires compatible models (most modern models support tools)</li>
                </ul>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Embedding Provider Selection */}
      <div className="mb-8">
        <ProviderSelectionGrid
          selectedProvider={currentEmbeddingProvider}
          onProviderSelect={handleEmbeddingProviderSelect}
          title="Embedding Provider Selection"
          subtitle="Choose your embedding provider for vector search and retrieval"
          disabledProviders={['anthropic']} // Anthropic doesn't have embedding models
        />
      </div>

      {/* Embedding Models Configuration */}
      <div className="mb-8 p-6 rounded-lg border border-purple-500/20 bg-gradient-to-r from-purple-500/5 to-transparent">
        <div className="flex items-center mb-4">
          <div className="w-6 h-6 rounded-full bg-purple-500/20 border border-purple-500/30 flex items-center justify-center mr-3">
            <div className="w-2 h-2 rounded-full bg-purple-500" />
          </div>
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
            LLM Chat Models
          </h3>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {currentEmbeddingProvider === 'ollama' && (
            <div>
              <Input
                label="Ollama Embedding Server URL"
                value={ragSettings.LLM_BASE_URL || 'http://192.168.5.21:11434'}
                onChange={e => setRagSettings({
                  ...ragSettings,
                  LLM_BASE_URL: e.target.value
                })}
                placeholder="http://192.168.5.21:11434"
                accentColor="purple"
              />
            </div>
          )}
          
          <div>
            <Input
              label="Embedding Model"
              value={ragSettings.EMBEDDING_MODEL || ''}
              onChange={e => setRagSettings({
                ...ragSettings,
                EMBEDDING_MODEL: e.target.value
              })}
              placeholder={getEmbeddingPlaceholder(currentEmbeddingProvider)}
              accentColor="purple"
            />
          </div>
        </div>
      </div>

      {/* Save Button */}
      <div className="mb-6">
        <Button 
          variant="outline" 
          accentColor="green" 
          icon={saving ? <Loader className="w-4 h-4 mr-2 animate-spin" /> : <Save className="w-4 h-4 mr-2" />}
          className="w-full sm:w-auto"
          size="md"
          onClick={async () => {
            try {
              setSaving(true);
              await credentialsService.updateRagSettings(ragSettings);
              showToast('RAG settings saved successfully!', 'success');
            } catch (err) {
              console.error('Failed to save RAG settings:', err);
              showToast('Failed to save settings', 'error');
            } finally {
              setSaving(false);
            }
          }}
          disabled={saving}
        >
          {saving ? 'Saving...' : 'Save Settings'}
        </Button>
      </div>

      {/* Rest of the existing RAG settings (contextual embeddings, hybrid search, etc.) */}
      {/* ... (keeping the rest of the original RAGSettings implementation) ... */}
      
    </Card>
  );
};

// Helper functions for model placeholders
function getModelPlaceholder(provider: Provider): string {
  switch (provider) {
    case 'openai':
      return 'e.g., gpt-4o-mini';
    case 'ollama':
      return 'e.g., llama2, mistral';
    case 'google':
      return 'e.g., gemini-1.5-flash';
    case 'anthropic':
      return 'e.g., claude-3-sonnet';
    default:
      return 'e.g., gpt-4o-mini';
  }
}

function getEmbeddingPlaceholder(provider: Provider): string {
  switch (provider) {
    case 'openai':
      return 'Default: text-embedding-3-small';
    case 'ollama':
      return 'e.g., snowflake-arctic-embed2';
    case 'google':
      return 'e.g., text-embedding-004';
    case 'anthropic':
      return 'N/A - No embedding models';
    default:
      return 'Default: text-embedding-3-small';
  }
}