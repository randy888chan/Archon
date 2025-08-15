import React, { useState, useEffect, useMemo } from 'react';
import { X, Search, Activity, Cpu, Database, Zap, Clock, Star, Download, Loader } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { createPortal } from 'react-dom';
import { Button } from '../ui/Button';
import { Input } from '../ui/Input';
import { Badge } from '../ui/Badge';
import { Provider } from './ProviderTileButton';

export interface ModelSpec {
  id: string;
  name: string;
  displayName: string;
  provider: Provider;
  type: 'chat' | 'embedding' | 'vision';
  description?: string;
  contextWindow?: number;
  recommended?: boolean;
  dimensions?: number;
  toolSupport?: boolean;
  performance?: { speed: 'fast' | 'medium' | 'slow'; quality: 'high' | 'medium' | 'low' };
  capabilities?: string[];
  useCase?: string[];
  status?: 'available' | 'downloading' | 'error';
  size_gb?: number;
  family?: string;
  hostInfo?: {
    host: string;
    family?: string;
    size_gb?: number;
  };
  pricing?: {
    input: number;
    output: number;
    unit: string;
  };
}

interface ModelSelectionModalProps {
  isOpen: boolean;
  onClose: () => void;
  provider: Provider;
  modelType: 'chat' | 'embedding';
  onSelectModel: (model: ModelSpec) => void;
  selectedModelId?: string;
  loading?: boolean;
}

type SortOption = 'name' | 'contextWindow' | 'performance' | 'pricing';
type SortDirection = 'asc' | 'desc';

const getMockModels = (provider: Provider): ModelSpec[] => {
  const models: Record<Provider, ModelSpec[]> = {
    openai: [
      {
        id: 'gpt-4-turbo',
        name: 'gpt-4-turbo',
        displayName: 'GPT-4 Turbo',
        provider: 'openai',
        type: 'chat',
        description: 'Most capable GPT-4 model with improved instruction following',
        contextWindow: 128000,
        recommended: true,
        toolSupport: true,
        performance: { speed: 'medium', quality: 'high' },
        capabilities: ['Text Generation', 'Function Calling', 'Code Generation'],
        useCase: ['General Purpose', 'Complex Reasoning'],
        status: 'available',
        pricing: { input: 0.01, output: 0.03, unit: '1K tokens' }
      },
      {
        id: 'gpt-4',
        name: 'gpt-4',
        displayName: 'GPT-4',
        provider: 'openai',
        type: 'chat',
        description: 'High-quality reasoning and complex instruction following',
        contextWindow: 8192,
        toolSupport: true,
        performance: { speed: 'slow', quality: 'high' },
        capabilities: ['Text Generation', 'Function Calling'],
        useCase: ['General Purpose'],
        status: 'available',
        pricing: { input: 0.03, output: 0.06, unit: '1K tokens' }
      },
      {
        id: 'text-embedding-3-large',
        name: 'text-embedding-3-large',
        displayName: 'Text Embedding 3 Large',
        provider: 'openai',
        type: 'embedding',
        description: 'Most capable embedding model for semantic search',
        contextWindow: 8191,
        dimensions: 3072,
        recommended: true,
        performance: { speed: 'fast', quality: 'high' },
        capabilities: ['Text Embeddings', 'Semantic Search'],
        useCase: ['RAG', 'Search'],
        status: 'available',
        pricing: { input: 0.00013, output: 0, unit: '1K tokens' }
      },
      {
        id: 'text-embedding-3-small',
        name: 'text-embedding-3-small',
        displayName: 'Text Embedding 3 Small',
        provider: 'openai',
        type: 'embedding',
        description: 'Efficient embedding model for most use cases',
        contextWindow: 8191,
        dimensions: 1536,
        performance: { speed: 'fast', quality: 'medium' },
        capabilities: ['Text Embeddings'],
        useCase: ['RAG'],
        status: 'available',
        pricing: { input: 0.00002, output: 0, unit: '1K tokens' }
      }
    ],
    google: [
      {
        id: 'gemini-1.5-pro',
        name: 'gemini-1.5-pro',
        displayName: 'Gemini 1.5 Pro',
        provider: 'google',
        type: 'chat',
        description: 'Google\'s most capable multimodal model',
        contextWindow: 2000000,
        recommended: true,
        toolSupport: true,
        performance: { speed: 'medium', quality: 'high' },
        capabilities: ['Text Generation', 'Vision', 'Function Calling'],
        useCase: ['General Purpose', 'Multimodal'],
        status: 'available'
      },
      {
        id: 'gemini-1.5-flash',
        name: 'gemini-1.5-flash',
        displayName: 'Gemini 1.5 Flash',
        provider: 'google',
        type: 'chat',
        description: 'Fast and efficient with good performance',
        contextWindow: 1000000,
        toolSupport: true,
        performance: { speed: 'fast', quality: 'medium' },
        capabilities: ['Text Generation', 'Function Calling'],
        useCase: ['General Purpose'],
        status: 'available'
      }
    ],
    ollama: [],
    anthropic: [
      {
        id: 'claude-3-5-sonnet-20241022',
        name: 'claude-3-5-sonnet-20241022',
        displayName: 'Claude 3.5 Sonnet',
        provider: 'anthropic',
        type: 'chat',
        description: 'Anthropic\'s most intelligent model',
        contextWindow: 200000,
        recommended: true,
        toolSupport: true,
        performance: { speed: 'medium', quality: 'high' },
        capabilities: ['Text Generation', 'Function Calling', 'Code Generation'],
        useCase: ['General Purpose', 'Complex Reasoning'],
        status: 'available'
      },
      {
        id: 'claude-3-haiku-20240307',
        name: 'claude-3-haiku-20240307',
        displayName: 'Claude 3 Haiku',
        provider: 'anthropic',
        type: 'chat',
        description: 'Fast and cost-effective for lighter tasks',
        contextWindow: 200000,
        toolSupport: true,
        performance: { speed: 'fast', quality: 'medium' },
        capabilities: ['Text Generation', 'Function Calling'],
        useCase: ['General Purpose'],
        status: 'available'
      }
    ]
  };
  
  return models[provider] || [];
};

export const ModelSelectionModal: React.FC<ModelSelectionModalProps> = ({
  isOpen,
  onClose,
  provider,
  modelType,
  onSelectModel,
  selectedModelId,
  loading = false
}) => {
  const [searchQuery, setSearchQuery] = useState('');
  const [filterType, setFilterType] = useState<'all' | 'chat' | 'embedding' | 'vision'>('all');
  const [sortBy, setSortBy] = useState<SortOption>('name');
  const [sortDirection, setSortDirection] = useState<SortDirection>('asc');
  const [models, setModels] = useState<ModelSpec[]>([]);
  const [loadingModels, setLoadingModels] = useState(false);
  const [ollamaDiscovery, setOllamaDiscovery] = useState<{
    chat_models: any[];
    embedding_models: any[];
    host_status: Record<string, any>;
    total_models: number;
    discovery_errors: string[];
  } | null>(null);
  const [refreshKey, setRefreshKey] = useState(0);

  // Load models when modal opens
  useEffect(() => {
    if (isOpen) {
      loadModels();
    }
  }, [isOpen, provider, refreshKey]);

  // Filter models based on type preference
  useEffect(() => {
    if (modelType) {
      setFilterType(modelType);
    } else {
      setFilterType('all');
    }
  }, [modelType]);

  // Handle escape key
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    if (isOpen) {
      window.addEventListener('keydown', handleKeyDown);
    }
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose]);

  const loadModels = async () => {
    setLoadingModels(true);
    try {
      if (provider === 'ollama') {
        // For Ollama, use the discovery endpoint to get models from all hosts
        // First get the configured hosts (for now, use default)
        const hosts = ['http://localhost:11434']; // This should come from settings
        
        const discovery = await fetch('/api/providers/ollama/models', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            hosts,
            timeout_seconds: 10,
          }),
        });

        if (discovery.ok) {
          const discoveryData = await discovery.json();
          setOllamaDiscovery(discoveryData);
          
          // Convert Ollama models to ModelSpec format
          const allOllamaModels = [
            ...discoveryData.chat_models.map((model: any) => ({
              id: `${model.name}@${model.host}`,
              name: model.name,
              displayName: model.name,
              provider: 'ollama' as Provider,
              type: 'chat' as const,
              contextWindow: model.context_window,
              toolSupport: model.supports_tools,
              performance: { speed: 'medium', quality: 'high' },
              capabilities: ['Text Generation', ...(model.supports_tools ? ['Function Calling'] : [])],
              useCase: ['Local AI', 'Privacy'],
              status: 'available' as const,
              description: `${model.family || 'Ollama'} model on ${model.host}${model.size_gb ? ` (${model.size_gb}GB)` : ''}`,
              size_gb: model.size_gb,
              family: model.family,
              hostInfo: {
                host: model.host,
                family: model.family,
                size_gb: model.size_gb,
              },
            })),
            ...discoveryData.embedding_models.map((model: any) => ({
              id: `${model.name}@${model.host}`,
              name: model.name,
              displayName: model.name,
              provider: 'ollama' as Provider,
              type: 'embedding' as const,
              contextWindow: model.context_window,
              dimensions: model.embedding_dimensions,
              toolSupport: false,
              performance: { speed: 'fast', quality: 'medium' },
              capabilities: ['Text Embeddings', 'Local Processing'],
              useCase: ['Private Search', 'Local RAG'],
              status: 'available' as const,
              description: `${model.family || 'Embedding'} model on ${model.host}${model.size_gb ? ` (${model.size_gb}GB)` : ''}`,
              size_gb: model.size_gb,
              family: model.family,
              hostInfo: {
                host: model.host,
                family: model.family,
                size_gb: model.size_gb,
              },
            })),
          ];

          setModels(allOllamaModels);
        } else {
          throw new Error('Failed to discover Ollama models');
        }
      } else {
        // For other providers, use mock data since we don't have API endpoints yet
        setModels(getMockModels(provider));
      }
    } catch (error) {
      console.error('Error loading models:', error);
      // Fall back to mock data if API fails
      setModels(getMockModels(provider));
    } finally {
      setLoadingModels(false);
    }
  };

  const handleRefresh = () => {
    setRefreshKey(prev => prev + 1);
  };

  // Filter and sort models
  const filteredAndSortedModels = useMemo(() => {
    let filtered = models.filter(model => {
      // Text search filter
      const matchesSearch = !searchQuery || 
        model.displayName.toLowerCase().includes(searchQuery.toLowerCase()) ||
        model.description?.toLowerCase().includes(searchQuery.toLowerCase()) ||
        model.capabilities?.some(cap => cap.toLowerCase().includes(searchQuery.toLowerCase()));

      // Type filter
      const matchesType = filterType === 'all' || model.type === filterType;

      return matchesSearch && matchesType;
    });

    // Sort models
    filtered.sort((a, b) => {
      let aVal: any, bVal: any;
      
      switch (sortBy) {
        case 'name':
          aVal = a.displayName.toLowerCase();
          bVal = b.displayName.toLowerCase();
          break;
        case 'contextWindow':
          aVal = a.contextWindow || 0;
          bVal = b.contextWindow || 0;
          break;
        case 'performance':
          const speedOrder = { fast: 3, medium: 2, slow: 1 };
          const qualityOrder = { high: 3, medium: 2, low: 1 };
          aVal = speedOrder[a.performance?.speed || 'medium'] + qualityOrder[a.performance?.quality || 'medium'];
          bVal = speedOrder[b.performance?.speed || 'medium'] + qualityOrder[b.performance?.quality || 'medium'];
          break;
        case 'pricing':
          aVal = a.pricing?.input || 0;
          bVal = b.pricing?.input || 0;
          break;
        default:
          aVal = a.displayName;
          bVal = b.displayName;
      }

      if (sortDirection === 'asc') {
        return aVal < bVal ? -1 : aVal > bVal ? 1 : 0;
      } else {
        return aVal > bVal ? -1 : aVal < bVal ? 1 : 0;
      }
    });

    // Put recommended models first
    filtered.sort((a, b) => {
      if (a.recommended && !b.recommended) return -1;
      if (!a.recommended && b.recommended) return 1;
      return 0;
    });

    return filtered;
  }, [models, searchQuery, filterType, sortBy, sortDirection]);

  const handleSort = (option: SortOption) => {
    if (sortBy === option) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortBy(option);
      setSortDirection('asc');
    }
  };

  if (!isOpen) return null;

  return createPortal(
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 flex items-center justify-center z-50 bg-black/60 backdrop-blur-sm p-4"
      onClick={onClose}
    >
      <motion.div
        initial={{ scale: 0.9, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        exit={{ scale: 0.9, opacity: 0 }}
        className="relative bg-gray-900/95 border border-gray-800 rounded-xl w-full max-w-6xl h-[85vh] flex flex-col overflow-hidden shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header with gradient accent line */}
        <div className="absolute top-0 left-0 right-0 h-[2px] bg-gradient-to-r from-green-500 via-blue-500 via-orange-500 to-purple-500 shadow-[0_0_20px_5px_rgba(59,130,246,0.5)]"></div>
        
        {/* Modal Header */}
        <div className="flex justify-between items-center p-6 border-b border-gray-800">
          <div className="flex-1">
            <h2 className="text-2xl font-bold text-blue-400 flex items-center gap-3">
              <Activity className="w-6 h-6" />
              Select {provider.charAt(0).toUpperCase() + provider.slice(1)} Model
            </h2>
            <p className="text-gray-400 mt-1">
              Choose the best model for your needs
              {modelType && <span className="ml-1">({modelType} models)</span>}
            </p>
          </div>
          <div className="flex items-center gap-3">
            {provider === 'ollama' && (
              <Button
                variant="outline"
                size="sm"
                onClick={handleRefresh}
                disabled={loadingModels}
                className="text-orange-400 border-orange-500/30 hover:bg-orange-500/10"
              >
                {loadingModels ? <Loader className="w-4 h-4 animate-spin" /> : <Download className="w-4 h-4" />}
                {loadingModels ? 'Loading...' : 'Refresh'}
              </Button>
            )}
            <Button
              variant="ghost"
              size="sm"
              onClick={onClose}
              className="text-gray-400 hover:text-white hover:bg-gray-800"
            >
              <X className="w-5 h-5" />
            </Button>
          </div>
        </div>

        {/* Search and Filters */}
        <div className="p-4 border-b border-gray-800 space-y-4">
          <div className="flex gap-4">
            <div className="flex-1 relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
              <Input
                type="text"
                placeholder="Search models by name, description, or capabilities..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-10 bg-gray-800/50 border-gray-700 text-white placeholder-gray-400"
              />
            </div>
            <div className="flex gap-2">
              <Button
                variant={sortBy === 'name' ? 'solid' : 'outline'}
                size="sm"
                onClick={() => handleSort('name')}
                className="text-xs"
              >
                Name {sortBy === 'name' && (sortDirection === 'asc' ? '↑' : '↓')}
              </Button>
              <Button
                variant={sortBy === 'contextWindow' ? 'solid' : 'outline'}
                size="sm"
                onClick={() => handleSort('contextWindow')}
                className="text-xs"
              >
                Context {sortBy === 'contextWindow' && (sortDirection === 'asc' ? '↑' : '↓')}
              </Button>
              <Button
                variant={sortBy === 'performance' ? 'solid' : 'outline'}
                size="sm"
                onClick={() => handleSort('performance')}
                className="text-xs"
              >
                Performance {sortBy === 'performance' && (sortDirection === 'asc' ? '↑' : '↓')}
              </Button>
            </div>
          </div>

          {/* Ollama Discovery Status */}
          {provider === 'ollama' && ollamaDiscovery && (
            <div className="bg-gray-800/30 rounded-lg p-3">
              <div className="flex items-center gap-4 text-sm">
                <div className="text-orange-400">
                  <Database className="w-4 h-4 inline mr-1" />
                  {ollamaDiscovery.total_models} models found
                </div>
                {ollamaDiscovery.discovery_errors.length > 0 && (
                  <div className="text-red-400">
                    {ollamaDiscovery.discovery_errors.length} connection errors
                  </div>
                )}
              </div>
            </div>
          )}
        </div>

        {/* Models Grid */}
        <div className="flex-1 overflow-y-auto p-6">
          {loadingModels ? (
            <div className="flex items-center justify-center h-full">
              <div className="text-center">
                <Loader className="w-8 h-8 animate-spin text-blue-400 mx-auto mb-4" />
                <p className="text-gray-400">Loading models...</p>
              </div>
            </div>
          ) : filteredAndSortedModels.length === 0 ? (
            <div className="flex items-center justify-center h-full">
              <div className="text-center">
                <Database className="w-16 h-16 text-gray-600 mx-auto mb-4" />
                <h3 className="text-xl font-medium text-gray-400 mb-2">No Models Found</h3>
                <p className="text-gray-500">
                  {provider === 'ollama' 
                    ? 'No Ollama models are available. Make sure Ollama is running and has models installed.'
                    : `No ${modelType} models available for ${provider}`
                  }
                </p>
                {provider === 'ollama' && (
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={handleRefresh}
                    className="mt-4 text-orange-400 border-orange-500/30"
                  >
                    <Download className="w-4 h-4 mr-2" />
                    Refresh Models
                  </Button>
                )}
              </div>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {filteredAndSortedModels.map((model) => (
                <motion.div
                  key={model.id}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  className={`relative group cursor-pointer rounded-xl border-2 transition-all duration-300 hover:scale-[1.02] ${
                    selectedModelId === model.id
                      ? 'border-blue-500 bg-blue-500/10 shadow-[0_0_20px_rgba(59,130,246,0.3)]'
                      : 'border-gray-700 bg-gray-800/50 hover:border-gray-600 hover:bg-gray-800/70'
                  }`}
                  onClick={() => onSelectModel(model)}
                >
                  {/* Recommended Badge */}
                  {model.recommended && (
                    <div className="absolute -top-2 -right-2 z-10">
                      <div className="bg-gradient-to-r from-yellow-400 to-orange-500 text-black text-xs font-bold px-2 py-1 rounded-full flex items-center gap-1">
                        <Star className="w-3 h-3" />
                        Recommended
                      </div>
                    </div>
                  )}

                  <div className="p-5">
                    {/* Header */}
                    <div className="flex items-start justify-between mb-3">
                      <div className="flex-1">
                        <h3 className="font-bold text-lg text-white mb-1 line-clamp-1">
                          {model.displayName}
                        </h3>
                        <div className="flex items-center gap-2 mb-2">
                          <Badge 
                            variant="outline" 
                            className={`text-xs ${
                              model.type === 'chat' ? 'border-green-500/30 text-green-400' :
                              model.type === 'embedding' ? 'border-purple-500/30 text-purple-400' :
                              'border-blue-500/30 text-blue-400'
                            }`}
                          >
                            {model.type}
                          </Badge>
                          {model.hostInfo?.host && (
                            <Badge variant="outline" className="text-xs border-orange-500/30 text-orange-400">
                              {new URL(model.hostInfo.host).hostname}
                            </Badge>
                          )}
                        </div>
                      </div>
                    </div>

                    {/* Description */}
                    {model.description && (
                      <p className="text-sm text-gray-400 mb-4 line-clamp-2">
                        {model.description}
                      </p>
                    )}

                    {/* Specs */}
                    <div className="space-y-3">
                      <div className="grid grid-cols-2 gap-3 text-xs">
                        {model.contextWindow && (
                          <div className="flex items-center gap-1 text-cyan-400">
                            <Cpu className="w-3 h-3" />
                            <span>{model.contextWindow.toLocaleString()} tokens</span>
                          </div>
                        )}
                        {model.dimensions && (
                          <div className="flex items-center gap-1 text-purple-400">
                            <Database className="w-3 h-3" />
                            <span>{model.dimensions}D</span>
                          </div>
                        )}
                        {model.performance && (
                          <div className="flex items-center gap-1 text-green-400">
                            <Zap className="w-3 h-3" />
                            <span>{model.performance.speed}</span>
                          </div>
                        )}
                        {model.size_gb && (
                          <div className="flex items-center gap-1 text-orange-400">
                            <Download className="w-3 h-3" />
                            <span>{model.size_gb}GB</span>
                          </div>
                        )}
                      </div>

                      {/* Capabilities */}
                      {model.capabilities && model.capabilities.length > 0 && (
                        <div className="flex flex-wrap gap-1">
                          {model.capabilities.slice(0, 3).map((cap, index) => (
                            <Badge
                              key={index}
                              variant="outline"
                              className="text-xs bg-gray-700/50 border-gray-600 text-gray-300"
                            >
                              {cap}
                            </Badge>
                          ))}
                          {model.capabilities.length > 3 && (
                            <Badge variant="outline" className="text-xs bg-gray-700/50 border-gray-600 text-gray-300">
                              +{model.capabilities.length - 3}
                            </Badge>
                          )}
                        </div>
                      )}

                      {/* Pricing */}
                      {model.pricing && (
                        <div className="text-xs text-gray-400">
                          <span className="text-green-400">${model.pricing.input}</span>
                          /<span className="text-blue-400">${model.pricing.output}</span> per {model.pricing.unit}
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Selected indicator */}
                  {selectedModelId === model.id && (
                    <div className="absolute inset-0 border-2 border-blue-500 rounded-xl pointer-events-none">
                      <div className="absolute top-3 right-3 w-6 h-6 bg-blue-500 rounded-full flex items-center justify-center">
                        <div className="w-2 h-2 bg-white rounded-full" />
                      </div>
                    </div>
                  )}
                </motion.div>
              ))}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="flex justify-between items-center p-6 border-t border-gray-800">
          <div className="text-sm text-gray-400">
            {filteredAndSortedModels.length} model{filteredAndSortedModels.length !== 1 ? 's' : ''} available
          </div>
          <div className="flex gap-3">
            <Button variant="outline" onClick={onClose}>
              Cancel
            </Button>
          </div>
        </div>
      </motion.div>
    </motion.div>,
    document.body
  );
};