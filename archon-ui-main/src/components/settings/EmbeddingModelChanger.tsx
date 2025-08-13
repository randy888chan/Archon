import React, { useState, useEffect } from 'react';
import { AlertTriangle, Trash2, RefreshCw, CheckCircle, Info } from 'lucide-react';
import { Card } from '../ui/Card';
import { Button } from '../ui/Button';
import { Input } from '../ui/Input';
import { Select } from '../ui/Select';
import { useToast } from '../../contexts/ToastContext';

interface ModelRecommendation {
  model_name: string;
  provider: string;
  dimensions: number;
  description: string;
  use_case: string;
}

interface ModelValidation {
  is_valid: boolean;
  is_change: boolean;
  dimensions_change: boolean;
  requires_migration: boolean;
  data_loss_warning: boolean;
  current: {
    provider: string;
    model: string;
    dimensions: number;
  };
  new: {
    provider: string;
    model: string;
    dimensions: number;
  };
  error?: string;
}

interface CurrentModelInfo {
  provider: string;
  model_name: string;
  dimensions: number;
  schema_dimensions: Record<string, number>;
  embedding_counts: Record<string, number>;
  total_embeddings: number;
  schema_needs_migration: boolean;
}

export const EmbeddingModelChanger = () => {
  const [recommendations, setRecommendations] = useState<ModelRecommendation[]>([]);
  const [currentModel, setCurrentModel] = useState<CurrentModelInfo | null>(null);
  const [selectedProvider, setSelectedProvider] = useState('');
  const [selectedModel, setSelectedModel] = useState('');
  const [validation, setValidation] = useState<ModelValidation | null>(null);
  const [showWarning, setShowWarning] = useState(false);
  const [confirmationText, setConfirmationText] = useState('');
  const [isChanging, setIsChanging] = useState(false);
  const [loading, setLoading] = useState(true);
  const { showToast } = useToast();

  const requiredConfirmation = "I understand this will permanently delete all embeddings";

  useEffect(() => {
    loadInitialData();
  }, []);

  const loadInitialData = async () => {
    try {
      setLoading(true);
      
      // Load recommendations and current model info in parallel
      const [recsResponse, currentResponse] = await Promise.all([
        fetch('/api/embedding-models/recommendations'),
        fetch('/api/embedding-models/current')
      ]);

      if (recsResponse.ok) {
        const recs = await recsResponse.json();
        setRecommendations(recs);
      }

      if (currentResponse.ok) {
        const current = await currentResponse.json();
        setCurrentModel(current);
        setSelectedProvider(current.provider);
        setSelectedModel(current.model_name);
      }
    } catch (error) {
      console.error('Failed to load initial data:', error);
      showToast('Failed to load embedding model information', 'error');
    } finally {
      setLoading(false);
    }
  };

  const validateModelChange = async () => {
    if (!selectedProvider || !selectedModel) {
      showToast('Please select a provider and model', 'error');
      return;
    }

    try {
      const response = await fetch('/api/embedding-models/validate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          provider: selectedProvider,
          model_name: selectedModel
        })
      });

      if (response.ok) {
        const validationResult = await response.json();
        setValidation(validationResult);
        
        if (validationResult.data_loss_warning) {
          setShowWarning(true);
        } else if (validationResult.is_change) {
          // No data loss warning, proceed directly
          await performModelChange();
        } else {
          showToast('Selected model is the same as current model', 'info');
        }
      } else {
        showToast('Failed to validate model change', 'error');
      }
    } catch (error) {
      console.error('Validation failed:', error);
      showToast('Failed to validate model change', 'error');
    }
  };

  const performModelChange = async () => {
    try {
      setIsChanging(true);
      
      const response = await fetch('/api/embedding-models/change', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          provider: selectedProvider,
          model_name: selectedModel
        })
      });

      if (response.ok) {
        const result = await response.json();
        showToast('Embedding model changed successfully!', 'success');
        
        // Reset state
        setShowWarning(false);
        setConfirmationText('');
        setValidation(null);
        
        // Reload current model info
        await loadInitialData();
      } else {
        showToast('Failed to change embedding model', 'error');
      }
    } catch (error) {
      console.error('Model change failed:', error);
      showToast('Failed to change embedding model', 'error');
    } finally {
      setIsChanging(false);
    }
  };

  const handleConfirmedChange = async () => {
    if (confirmationText !== requiredConfirmation) {
      showToast('Please enter the exact confirmation text', 'error');
      return;
    }
    
    await performModelChange();
  };

  const cancelChange = () => {
    setShowWarning(false);
    setConfirmationText('');
    setValidation(null);
  };

  if (loading) {
    return (
      <Card accentColor="blue" className="p-6">
        <div className="flex items-center justify-center py-8">
          <RefreshCw className="w-6 h-6 animate-spin text-blue-500 mr-3" />
          <span className="text-gray-600 dark:text-gray-300">Loading embedding model information...</span>
        </div>
      </Card>
    );
  }

  // Group recommendations by provider for easier selection
  const providerOptions = [...new Set(recommendations.map(r => r.provider))].map(provider => ({
    value: provider,
    label: provider.charAt(0).toUpperCase() + provider.slice(1)
  }));

  const modelOptions = recommendations
    .filter(r => r.provider === selectedProvider)
    .map(r => ({
      value: r.model_name,
      label: `${r.model_name} (${r.dimensions}d)`
    }));

  const selectedModelInfo = recommendations.find(
    r => r.provider === selectedProvider && r.model_name === selectedModel
  );

  return (
    <div className="space-y-6">
      <Card accentColor="blue" className="p-6">
        <div className="flex items-center mb-4">
          <CheckCircle className="mr-2 text-blue-500 filter drop-shadow-[0_0_8px_rgba(59,130,246,0.8)]" size={20} />
          <h2 className="text-xl font-semibold text-gray-800 dark:text-white">
            Embedding Model Manager
          </h2>
        </div>

        <p className="text-sm text-gray-600 dark:text-zinc-400 mb-6">
          Manage your embedding model configuration. Changing models with different dimensions will require re-embedding existing content.
        </p>

        {currentModel && (
          <div className="bg-blue-500/5 border border-blue-500/20 rounded-lg p-4 mb-6">
            <h3 className="font-medium text-gray-800 dark:text-white mb-2">Current Model</h3>
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <span className="text-gray-600 dark:text-gray-400">Provider:</span>
                <span className="ml-2 font-medium text-gray-800 dark:text-white">
                  {currentModel.provider.charAt(0).toUpperCase() + currentModel.provider.slice(1)}
                </span>
              </div>
              <div>
                <span className="text-gray-600 dark:text-gray-400">Model:</span>
                <span className="ml-2 font-medium text-gray-800 dark:text-white">
                  {currentModel.model_name}
                </span>
              </div>
              <div>
                <span className="text-gray-600 dark:text-gray-400">Dimensions:</span>
                <span className="ml-2 font-medium text-gray-800 dark:text-white">
                  {currentModel.dimensions}
                </span>
              </div>
              <div>
                <span className="text-gray-600 dark:text-gray-400">Total Embeddings:</span>
                <span className="ml-2 font-medium text-gray-800 dark:text-white">
                  {currentModel.total_embeddings.toLocaleString()}
                </span>
              </div>
            </div>
          </div>
        )}

        <div className="grid grid-cols-2 gap-4 mb-6">
          <div>
            <Select
              label="Provider"
              value={selectedProvider}
              onChange={(e) => {
                setSelectedProvider(e.target.value);
                setSelectedModel(''); // Reset model when provider changes
              }}
              options={providerOptions}
              accentColor="blue"
            />
          </div>
          <div>
            <Select
              label="Model"
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              options={modelOptions}
              accentColor="blue"
              disabled={!selectedProvider}
            />
          </div>
        </div>

        {selectedModelInfo && (
          <div className="bg-gray-50 dark:bg-gray-800/50 rounded-lg p-4 mb-6">
            <h4 className="font-medium text-gray-800 dark:text-white mb-2">
              {selectedModelInfo.model_name}
            </h4>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
              {selectedModelInfo.description}
            </p>
            <p className="text-xs text-gray-500 dark:text-gray-500">
              <strong>Use case:</strong> {selectedModelInfo.use_case}
            </p>
            <p className="text-xs text-gray-500 dark:text-gray-500">
              <strong>Dimensions:</strong> {selectedModelInfo.dimensions}
            </p>
          </div>
        )}

        <div className="flex justify-end">
          <Button
            onClick={validateModelChange}
            disabled={!selectedProvider || !selectedModel}
            accentColor="blue"
            className="px-6"
          >
            Change Model
          </Button>
        </div>
      </Card>

      {showWarning && validation && (
        <Card accentColor="red" className="p-6">
          <div className="flex items-center mb-4">
            <AlertTriangle className="mr-2 text-red-500 filter drop-shadow-[0_0_8px_rgba(239,68,68,0.8)]" size={20} />
            <h3 className="text-lg font-semibold text-red-800 dark:text-red-300">
              Warning: Data Loss Risk
            </h3>
          </div>

          <div className="space-y-4">
            <p className="text-sm text-red-700 dark:text-red-300">
              Changing from <strong>{validation.current.model}</strong> ({validation.current.dimensions}d) 
              to <strong>{validation.new.model}</strong> ({validation.new.dimensions}d) will:
            </p>

            <ul className="list-disc pl-6 space-y-1 text-sm text-red-700 dark:text-red-300">
              {validation.dimensions_change && (
                <li>Change vector dimensions from {validation.current.dimensions} to {validation.new.dimensions}</li>
              )}
              {validation.requires_migration && (
                <li>Delete all existing embeddings ({currentModel?.total_embeddings.toLocaleString()} items)</li>
              )}
              <li>Require re-embedding of all content (this may take time and cost money)</li>
              <li>Temporarily reduce search quality until re-embedding is complete</li>
            </ul>

            <div className="bg-red-500/10 border border-red-500/20 rounded-lg p-4">
              <div className="flex items-start">
                <Info className="w-4 h-4 text-red-500 mr-2 mt-0.5 flex-shrink-0" />
                <div>
                  <p className="text-sm text-red-700 dark:text-red-300 font-medium mb-1">
                    This action cannot be undone
                  </p>
                  <p className="text-xs text-red-600 dark:text-red-400">
                    Please ensure you have a backup if you want to revert this change.
                  </p>
                </div>
              </div>
            </div>

            <div>
              <label className="block text-sm font-medium text-red-700 dark:text-red-300 mb-2">
                Type the following text to confirm:
              </label>
              <Input
                value={confirmationText}
                onChange={(e) => setConfirmationText(e.target.value)}
                placeholder={requiredConfirmation}
                className="mb-4"
                accentColor="red"
              />
              <p className="text-xs text-red-600 dark:text-red-400">
                Required: "{requiredConfirmation}"
              </p>
            </div>

            <div className="flex justify-end space-x-3">
              <Button
                onClick={cancelChange}
                variant="outline"
                accentColor="gray"
              >
                Cancel
              </Button>
              <Button
                onClick={handleConfirmedChange}
                disabled={confirmationText !== requiredConfirmation || isChanging}
                accentColor="red"
                icon={isChanging ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Trash2 className="w-4 h-4" />}
              >
                {isChanging ? 'Changing Model...' : 'Confirm Change'}
              </Button>
            </div>
          </div>
        </Card>
      )}
    </div>
  );
};