import React, { useState, useRef, useEffect, useCallback } from 'react';
import { Settings, Check, Save, Loader, ChevronDown, ChevronUp, Zap, Database, MessageSquare, Layers } from 'lucide-react';
import { Card } from '../ui/Card';
import { Input } from '../ui/Input';
import { Select } from '../ui/Select';
import { Button } from '../ui/Button';
import { Slider } from '../ui/Slider';
import { Switch } from '../ui/Switch';
import { Collapsible, CollapsibleTrigger, CollapsibleContent } from '../ui/Collapsible';
import { useToast } from '../../contexts/ToastContext';
import { credentialsService } from '../../services/credentialsService';
import OllamaConfigurationPanel from './OllamaConfigurationPanel';
import { ProviderSelectionGrid } from './ProviderSelectionGrid';
import { ModelSelectionModal, ModelSpec } from './ModelSelectionModal';
import { Provider } from './ProviderTileButton';

interface RAGSettingsInterface {
  LLM_PROVIDER: string;
  MODEL_CHOICE: string;
  EMBEDDING_MODEL: string;
  TEMPERATURE?: number;
  USE_CONTEXTUAL_EMBEDDINGS?: boolean;
  API_KEY?: string;
  GEMINI_API_KEY?: string;
  ANTHROPIC_API_KEY?: string;
  LLM_BASE_URL?: string;
  MAX_PAGES?: number;
  MAX_DEPTH?: number;
  CHUNK_SIZE?: number;
  CHUNK_OVERLAP?: number;
  // Add fields from RagSettings interface to make them compatible
  USE_HYBRID_SEARCH?: boolean;
  USE_AGENTIC_RAG?: boolean;
  USE_RERANKING?: boolean;
  CONTEXTUAL_EMBEDDINGS_MAX_WORKERS?: number;
  CRAWL_BATCH_SIZE?: number;
  CRAWL_MAX_CONCURRENT?: number;
  CRAWL_WAIT_STRATEGY?: string;
  CRAWL_PAGE_TIMEOUT?: number;
  CRAWL_DELAY_BEFORE_HTML?: number;
  DOCUMENT_STORAGE_BATCH_SIZE?: number;
  EMBEDDING_BATCH_SIZE?: number;
  DELETE_BATCH_SIZE?: number;
  ENABLE_PARALLEL_BATCHES?: boolean;
  MEMORY_THRESHOLD_PERCENT?: number;
  DISPATCHER_CHECK_INTERVAL?: number;
  CODE_EXTRACTION_BATCH_SIZE?: number;
  CODE_SUMMARY_MAX_WORKERS?: number;
}

interface RAGSettingsProps {
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

export const RAGSettings = (props: {
  ragSettings: RAGSettingsInterface;
  setRagSettings: (settings: RAGSettingsInterface) => void;
}) => {
  const { ragSettings, setRagSettings } = props;
  const [saving, setSaving] = useState(false);
  const [modelSelectionModalOpen, setModelSelectionModalOpen] = useState(false);
  const [embeddingModelSelectionModalOpen, setEmbeddingModelSelectionModalOpen] = useState(false);
  const [currentModelType, setCurrentModelType] = useState<'chat' | 'embedding'>('chat');
  const [showCrawlingSettings, setShowCrawlingSettings] = useState(false);
  const [showStorageSettings, setShowStorageSettings] = useState(false);
  const { showToast } = useToast();

  // Debounced save function to prevent API overload
  const debouncedSaveTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  
  const saveSettings = useCallback(async (newSettings: RAGSettingsInterface) => {
    try {
      setSaving(true);
      await credentialsService.updateRagSettings(newSettings as any);
      showToast('Settings saved successfully', 'success');
    } catch (error) {
      console.error('Failed to save settings:', error);
      showToast('Failed to save settings', 'error');
    } finally {
      setSaving(false);
    }
  }, [showToast]);

  // Debounced save with 1 second delay
  const debouncedSaveSettings = useCallback((newSettings: RAGSettingsInterface) => {
    // Clear existing timeout
    if (debouncedSaveTimeoutRef.current) {
      clearTimeout(debouncedSaveTimeoutRef.current);
    }
    
    // Set new timeout
    debouncedSaveTimeoutRef.current = setTimeout(() => {
      saveSettings(newSettings);
    }, 1000); // 1 second debounce
  }, [saveSettings]);

  // Cleanup timeout on unmount
  useEffect(() => {
    return () => {
      if (debouncedSaveTimeoutRef.current) {
        clearTimeout(debouncedSaveTimeoutRef.current);
      }
    };
  }, []);

  // Handle Ollama configuration changes
  const handleOllamaConfigChange = useCallback((instances: any[]) => {
    // Find primary instance for LLM_BASE_URL
    const primaryInstance = instances.find(inst => inst.isPrimary) || instances[0];
    
    if (primaryInstance) {
      setRagSettings(prevSettings => {
        const newSettings = {
          ...prevSettings,
          LLM_BASE_URL: primaryInstance.baseUrl
        };
        debouncedSaveSettings(newSettings); // Use debounced save
        return newSettings;
      });
    }
  }, [debouncedSaveSettings]);

  const handleProviderSelect = (provider: Provider) => {
    const newSettings = {
      ...ragSettings,
      LLM_PROVIDER: provider,
      // Reset model choices when provider changes
      MODEL_CHOICE: '',
      EMBEDDING_MODEL: ''
    };
    setRagSettings(newSettings);
    debouncedSaveSettings(newSettings); // Use debounced save
  };

  const handleModelSelection = (model: ModelSpec) => {
    if (currentModelType === 'chat') {
      const newSettings = {
        ...ragSettings,
        MODEL_CHOICE: model.name
      };
      setRagSettings(newSettings);
      debouncedSaveSettings(newSettings); // Use debounced save
      setModelSelectionModalOpen(false);
    } else {
      const newSettings = {
        ...ragSettings,
        EMBEDDING_MODEL: model.name
      };
      setRagSettings(newSettings);
      debouncedSaveSettings(newSettings); // Use debounced save
      setEmbeddingModelSelectionModalOpen(false);
    }
  };

  const openChatModelSelection = () => {
    setCurrentModelType('chat');
    setModelSelectionModalOpen(true);
  };

  const openEmbeddingModelSelection = () => {
    setCurrentModelType('embedding');
    setEmbeddingModelSelectionModalOpen(true);
  };

  // Safe provider name display
  const getProviderDisplayName = (provider: string | undefined) => {
    if (!provider) return 'Select Provider';
    return provider.charAt(0).toUpperCase() + provider.slice(1);
  };

  return (
    <div className="space-y-8">
      {/* Provider Selection */}
      <Card accentColor="green" className="p-6">
        <ProviderSelectionGrid
          selectedProvider={ragSettings.LLM_PROVIDER as Provider}
          onProviderSelect={handleProviderSelect}
          title="AI Provider"
          subtitle="Choose your preferred AI provider for chat completions"
        />
      </Card>

      {/* Ollama Configuration Panel */}
      <OllamaConfigurationPanel
        isVisible={ragSettings.LLM_PROVIDER === 'ollama'}
        onConfigChange={handleOllamaConfigChange}
      />

      {/* Model Configuration */}
      <Card accentColor="blue" className="p-6 space-y-6">
        <div className="flex items-center mb-4">
          <div className="w-6 h-6 rounded-full bg-blue-500/20 border border-blue-500/30 flex items-center justify-center mr-3">
            <div className="w-2 h-2 rounded-full bg-blue-500" />
          </div>
          <div>
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
              Model Selection
            </h3>
            <p className="text-sm text-gray-600 dark:text-gray-400">
              Choose specific models for chat and embeddings
            </p>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Chat Model Selection Button */}
          <div className="space-y-2">
            <label className="text-sm font-medium text-gray-900 dark:text-white">Chat Model</label>
            <Button
              variant="outline"
              className="w-full justify-between h-12 px-4"
              onClick={openChatModelSelection}
              disabled={!ragSettings.LLM_PROVIDER}
            >
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 rounded-full bg-gradient-to-br from-blue-500 to-purple-600 flex items-center justify-center">
                  <MessageSquare className="w-4 h-4 text-white" />
                </div>
                <div className="text-left">
                  <div className="text-sm font-medium text-gray-900 dark:text-white">
                    {ragSettings.MODEL_CHOICE || 'Select Chat Model'}
                  </div>
                  <div className="text-xs text-gray-500 dark:text-gray-400">
                    {getProviderDisplayName(ragSettings.LLM_PROVIDER)} Provider
                  </div>
                </div>
              </div>
              <ChevronDown className="w-4 h-4 text-gray-400" />
            </Button>
          </div>

          {/* Embedding Model Selection Button */}
          <div className="space-y-2">
            <label className="text-sm font-medium text-gray-900 dark:text-white">Embedding Model</label>
            <Button
              variant="outline"
              className="w-full justify-between h-12 px-4"
              onClick={openEmbeddingModelSelection}
              disabled={!ragSettings.LLM_PROVIDER}
            >
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 rounded-full bg-gradient-to-br from-green-500 to-teal-600 flex items-center justify-center">
                  <Layers className="w-4 h-4 text-white" />
                </div>
                <div className="text-left">
                  <div className="text-sm font-medium text-gray-900 dark:text-white">
                    {ragSettings.EMBEDDING_MODEL || 'Select Embedding Model'}
                  </div>
                  <div className="text-xs text-gray-500 dark:text-gray-400">
                    {getProviderDisplayName(ragSettings.LLM_PROVIDER)} Provider
                  </div>
                </div>
              </div>
              <ChevronDown className="w-4 h-4 text-gray-400" />
            </Button>
          </div>
        </div>
      </Card>

      {/* Model Selection Modals */}
      {ragSettings.LLM_PROVIDER && (
        <>
          <ModelSelectionModal
            isOpen={modelSelectionModalOpen}
            onClose={() => setModelSelectionModalOpen(false)}
            provider={ragSettings.LLM_PROVIDER as Provider}
            modelType="chat"
            onSelectModel={handleModelSelection}
            selectedModelId={ragSettings.MODEL_CHOICE}
          />

          <ModelSelectionModal
            isOpen={embeddingModelSelectionModalOpen}
            onClose={() => setEmbeddingModelSelectionModalOpen(false)}
            provider={ragSettings.LLM_PROVIDER as Provider}
            modelType="embedding"
            onSelectModel={handleModelSelection}
            selectedModelId={ragSettings.EMBEDDING_MODEL}
          />
        </>
      )}
    </div>
  );
};