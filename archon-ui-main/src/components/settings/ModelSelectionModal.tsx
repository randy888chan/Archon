import React, { useEffect, useState, useMemo } from 'react';
import { createPortal } from 'react-dom';
import { motion } from 'framer-motion';
import {
  X,
  Search,
  Filter,
  Zap,
  Eye,
  Layers,
  Activity,
  DollarSign,
  CheckCircle2,
  AlertCircle,
  Loader2,
  SortAsc,
  SortDesc,
  Info,
  Cpu,
  Database,
  Users,
  Clock
} from 'lucide-react';
import { Button } from '../ui/Button';
import { Badge } from '../ui/Badge';
import { Provider } from './ProviderTileButton';
import { ModelSpecificationCard } from './ModelSpecificationCard';

// Model specification interfaces
export interface ModelSpec {
  id: string;
  name: string;
  displayName: string;
  provider: Provider;
  type: 'chat' | 'embedding' | 'vision';
  contextWindow: number;
  dimensions?: number; // For embedding models
  toolSupport: boolean;
  pricing?: {
    input?: number;  // per token
    output?: number; // per token
    unit?: string;
  };
  performance: {
    speed: 'fast' | 'medium' | 'slow';
    quality: 'high' | 'medium' | 'low';
  };
  capabilities: string[];
  useCase: string[];
  status: 'available' | 'installing' | 'error' | 'unavailable';
  description: string;
  maxTokens?: number;
  recommended?: boolean;
}

interface ModelSelectionModalProps {
  isOpen: boolean;
  onClose: () => void;
  provider: Provider;
  modelType?: 'chat' | 'embedding' | 'vision';
  onSelectModel: (model: ModelSpec) => void;
  selectedModelId?: string;
  loading?: boolean;
}

// Mock data for demonstration - in real implementation, this would come from API
const getMockModels = (provider: Provider): ModelSpec[] => {
  switch (provider) {
    case 'openai':
      return [
        {
          id: 'gpt-4o',
          name: 'gpt-4o',
          displayName: 'GPT-4o',
          provider: 'openai',
          type: 'chat',
          contextWindow: 128000,
          toolSupport: true,
          pricing: { input: 0.005, output: 0.015, unit: '1K tokens' },
          performance: { speed: 'fast', quality: 'high' },
          capabilities: ['Text Generation', 'Code Generation', 'Function Calling', 'Vision'],
          useCase: ['General Purpose', 'Coding', 'Analysis'],
          status: 'available',
          description: 'Most capable model with multimodal abilities and tool use',
          maxTokens: 4096,
          recommended: true
        },
        {
          id: 'gpt-4o-mini',
          name: 'gpt-4o-mini',
          displayName: 'GPT-4o Mini',
          provider: 'openai',
          type: 'chat',
          contextWindow: 128000,
          toolSupport: true,
          pricing: { input: 0.00015, output: 0.0006, unit: '1K tokens' },
          performance: { speed: 'fast', quality: 'high' },
          capabilities: ['Text Generation', 'Code Generation', 'Function Calling'],
          useCase: ['Cost-Effective', 'High Volume'],
          status: 'available',
          description: 'Affordable and intelligent small model for fast, lightweight tasks',
          maxTokens: 16384
        },
        {
          id: 'text-embedding-3-large',
          name: 'text-embedding-3-large',
          displayName: 'Text Embedding 3 Large',
          provider: 'openai',
          type: 'embedding',
          contextWindow: 8191,
          dimensions: 3072,
          toolSupport: false,
          pricing: { input: 0.00013, unit: '1K tokens' },
          performance: { speed: 'fast', quality: 'high' },
          capabilities: ['Text Embeddings', 'Semantic Search'],
          useCase: ['Semantic Search', 'RAG', 'Similarity'],
          status: 'available',
          description: 'Most capable embedding model with 3072 dimensions',
        },
        {
          id: 'text-embedding-3-small',
          name: 'text-embedding-3-small',
          displayName: 'Text Embedding 3 Small',
          provider: 'openai',
          type: 'embedding',
          contextWindow: 8191,
          dimensions: 1536,
          toolSupport: false,
          pricing: { input: 0.00002, unit: '1K tokens' },
          performance: { speed: 'fast', quality: 'medium' },
          capabilities: ['Text Embeddings', 'Semantic Search'],
          useCase: ['Cost-Effective Search', 'Basic RAG'],
          status: 'available',
          description: 'Efficient embedding model with good performance-cost ratio',
        }
      ];
    case 'ollama':
      return [
        {
          id: 'llama3.1:8b',
          name: 'llama3.1:8b',
          displayName: 'Llama 3.1 8B',
          provider: 'ollama',
          type: 'chat',
          contextWindow: 131072,
          toolSupport: true,
          performance: { speed: 'medium', quality: 'high' },
          capabilities: ['Text Generation', 'Code Generation', 'Function Calling'],
          useCase: ['Local AI', 'Privacy', 'General Purpose'],
          status: 'available',
          description: 'Fast and capable local language model with tool support',
          maxTokens: 4096,
          recommended: true
        },
        {
          id: 'llama3.1:70b',
          name: 'llama3.1:70b',
          displayName: 'Llama 3.1 70B',
          provider: 'ollama',
          type: 'chat',
          contextWindow: 131072,
          toolSupport: true,
          performance: { speed: 'slow', quality: 'high' },
          capabilities: ['Text Generation', 'Code Generation', 'Function Calling', 'Reasoning'],
          useCase: ['Complex Reasoning', 'High Quality Output'],
          status: 'installing',
          description: 'Large, powerful model for complex tasks (requires significant resources)',
          maxTokens: 4096
        },
        {
          id: 'nomic-embed-text',
          name: 'nomic-embed-text',
          displayName: 'Nomic Embed Text',
          provider: 'ollama',
          type: 'embedding',
          contextWindow: 8192,
          dimensions: 768,
          toolSupport: false,
          performance: { speed: 'fast', quality: 'medium' },
          capabilities: ['Text Embeddings', 'Local Processing'],
          useCase: ['Private Search', 'Local RAG'],
          status: 'available',
          description: 'High-quality local embedding model for privacy-focused applications'
        },
        {
          id: 'mxbai-embed-large',
          name: 'mxbai-embed-large',
          displayName: 'MxBai Embed Large',
          provider: 'ollama',
          type: 'embedding',
          contextWindow: 8192,
          dimensions: 1024,
          toolSupport: false,
          performance: { speed: 'medium', quality: 'high' },
          capabilities: ['Text Embeddings', 'Multilingual'],
          useCase: ['High-Quality Search', 'Multilingual RAG'],
          status: 'unavailable',
          description: 'Large embedding model with superior quality (not installed)'
        }
      ];
    case 'google':
      return [
        {
          id: 'gemini-1.5-pro',
          name: 'gemini-1.5-pro',
          displayName: 'Gemini 1.5 Pro',
          provider: 'google',
          type: 'chat',
          contextWindow: 2097152,
          toolSupport: true,
          pricing: { input: 0.00125, output: 0.005, unit: '1K tokens' },
          performance: { speed: 'medium', quality: 'high' },
          capabilities: ['Text Generation', 'Code Generation', 'Function Calling', 'Vision', 'Long Context'],
          useCase: ['Long Documents', 'Complex Analysis', 'Multimodal'],
          status: 'available',
          description: 'Largest context window with excellent multimodal capabilities',
          maxTokens: 8192,
          recommended: true
        },
        {
          id: 'gemini-1.5-flash',
          name: 'gemini-1.5-flash',
          displayName: 'Gemini 1.5 Flash',
          provider: 'google',
          type: 'chat',
          contextWindow: 1048576,
          toolSupport: true,
          pricing: { input: 0.00075, output: 0.003, unit: '1K tokens' },
          performance: { speed: 'fast', quality: 'high' },
          capabilities: ['Text Generation', 'Code Generation', 'Function Calling'],
          useCase: ['Fast Responses', 'High Volume'],
          status: 'available',
          description: 'Optimized for speed while maintaining quality'
        }
      ];
    case 'anthropic':
      return [
        {
          id: 'claude-3.5-sonnet',
          name: 'claude-3-5-sonnet-20241022',
          displayName: 'Claude 3.5 Sonnet',
          provider: 'anthropic',
          type: 'chat',
          contextWindow: 200000,
          toolSupport: true,
          pricing: { input: 0.003, output: 0.015, unit: '1K tokens' },
          performance: { speed: 'medium', quality: 'high' },
          capabilities: ['Text Generation', 'Code Generation', 'Function Calling', 'Analysis'],
          useCase: ['Reasoning', 'Writing', 'Coding'],
          status: 'available',
          description: 'Excellent reasoning and coding capabilities with strong safety',
          maxTokens: 4096,
          recommended: true
        },
        {
          id: 'claude-3-haiku',
          name: 'claude-3-haiku-20240307',
          displayName: 'Claude 3 Haiku',
          provider: 'anthropic',
          type: 'chat',
          contextWindow: 200000,
          toolSupport: true,
          pricing: { input: 0.00025, output: 0.00125, unit: '1K tokens' },
          performance: { speed: 'fast', quality: 'medium' },
          capabilities: ['Text Generation', 'Function Calling'],
          useCase: ['Fast Responses', 'Cost-Effective'],
          status: 'available',
          description: 'Fast and affordable model for everyday tasks'
        }
      ];
    default:
      return [];
  }
};

type SortOption = 'name' | 'contextWindow' | 'performance' | 'pricing';
type SortDirection = 'asc' | 'desc';

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

  // Load models when modal opens
  useEffect(() => {
    if (isOpen) {
      setLoadingModels(true);
      // Simulate API call delay
      setTimeout(() => {
        setModels(getMockModels(provider));
        setLoadingModels(false);
      }, 1000);
    }
  }, [isOpen, provider]);

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

  // Filter and sort models
  const filteredAndSortedModels = useMemo(() => {
    let filtered = models.filter(model => {
      // Text search filter
      const matchesSearch = !searchQuery || 
        model.displayName.toLowerCase().includes(searchQuery.toLowerCase()) ||
        model.description.toLowerCase().includes(searchQuery.toLowerCase()) ||
        model.capabilities.some(cap => cap.toLowerCase().includes(searchQuery.toLowerCase()));

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
          aVal = a.contextWindow;
          bVal = b.contextWindow;
          break;
        case 'performance':
          const speedOrder = { fast: 3, medium: 2, slow: 1 };
          const qualityOrder = { high: 3, medium: 2, low: 1 };
          aVal = speedOrder[a.performance.speed] + qualityOrder[a.performance.quality];
          bVal = speedOrder[b.performance.speed] + qualityOrder[b.performance.quality];
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
          <button
            onClick={onClose}
            className="text-gray-500 hover:text-white bg-gray-900/50 border border-gray-800 rounded-full p-2 transition-colors ml-4"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Filters and Search */}
        <div className="p-4 border-b border-gray-800 bg-gray-950/50">
          <div className="flex flex-col sm:flex-row gap-4">
            {/* Search */}
            <div className="flex-1 relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-500" />
              <input
                type="text"
                placeholder="Search models by name, capabilities..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="w-full pl-10 pr-3 py-2 bg-gray-900/70 border border-gray-800 rounded-lg text-sm text-gray-300 placeholder-gray-600 focus:outline-none focus:border-blue-500/50 focus:ring-1 focus:ring-blue-500/20 transition-all"
              />
            </div>

            {/* Type Filter */}
            <div className="flex items-center gap-2">
              <Filter className="w-4 h-4 text-gray-500" />
              <select
                value={filterType}
                onChange={(e) => setFilterType(e.target.value as any)}
                className="bg-gray-900/70 border border-gray-800 rounded-lg px-3 py-2 text-sm text-gray-300 focus:outline-none focus:border-blue-500/50"
              >
                <option value="all">All Types</option>
                <option value="chat">Chat Models</option>
                <option value="embedding">Embedding Models</option>
                <option value="vision">Vision Models</option>
              </select>
            </div>

            {/* Sort Options */}
            <div className="flex gap-1">
              <button
                onClick={() => handleSort('name')}
                className={`px-3 py-2 rounded-lg text-xs font-medium transition-colors flex items-center gap-1 ${
                  sortBy === 'name'
                    ? 'bg-blue-500/20 text-blue-400 border border-blue-500/40'
                    : 'bg-gray-800/50 text-gray-400 hover:text-gray-200 border border-gray-700'
                }`}
              >
                Name
                {sortBy === 'name' && (
                  sortDirection === 'asc' ? <SortAsc className="w-3 h-3" /> : <SortDesc className="w-3 h-3" />
                )}
              </button>
              <button
                onClick={() => handleSort('contextWindow')}
                className={`px-3 py-2 rounded-lg text-xs font-medium transition-colors flex items-center gap-1 ${
                  sortBy === 'contextWindow'
                    ? 'bg-blue-500/20 text-blue-400 border border-blue-500/40'
                    : 'bg-gray-800/50 text-gray-400 hover:text-gray-200 border border-gray-700'
                }`}
              >
                Context
                {sortBy === 'contextWindow' && (
                  sortDirection === 'asc' ? <SortAsc className="w-3 h-3" /> : <SortDesc className="w-3 h-3" />
                )}
              </button>
            </div>
          </div>
        </div>

        {/* Model Grid */}
        <div className="flex-1 overflow-auto p-4">
          {loadingModels || loading ? (
            <div className="h-full flex items-center justify-center">
              <div className="text-center">
                <Loader2 className="w-12 h-12 text-blue-400 mx-auto mb-4 animate-spin" />
                <p className="text-gray-400">Loading available models...</p>
              </div>
            </div>
          ) : filteredAndSortedModels.length === 0 ? (
            <div className="h-full flex items-center justify-center">
              <div className="text-center">
                <Activity className="w-12 h-12 text-gray-600 mx-auto mb-4" />
                <p className="text-gray-400">No models found matching your criteria</p>
              </div>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
              {filteredAndSortedModels.map((model) => (
                <ModelSpecificationCard
                  key={model.id}
                  model={model}
                  isSelected={selectedModelId === model.id}
                  onSelect={onSelectModel}
                  loading={loading}
                />
              ))}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="p-4 border-t border-gray-800 bg-gray-950/50">
          <div className="flex justify-between items-center">
            <div className="flex items-center gap-2 text-sm text-gray-400">
              <Info className="w-4 h-4" />
              <span>{filteredAndSortedModels.length} model{filteredAndSortedModels.length !== 1 ? 's' : ''} available</span>
            </div>
            <div className="flex gap-2">
              <Button
                variant="ghost"
                size="sm"
                onClick={onClose}
              >
                Cancel
              </Button>
            </div>
          </div>
        </div>
      </motion.div>
    </motion.div>,
    document.body
  );
};