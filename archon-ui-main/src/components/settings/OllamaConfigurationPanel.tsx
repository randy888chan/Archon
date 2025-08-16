import React, { useState, useEffect } from 'react';
import { Card } from '../ui/Card';
import { Button } from '../ui/Button';
import { Input } from '../ui/Input';
import { Badge } from '../ui/Badge';
import { useToast } from '../../contexts/ToastContext';
import { cn } from '../../lib/utils';
import { credentialsService, OllamaInstance } from '../../services/credentialsService';

interface OllamaConfigurationPanelProps {
  isVisible: boolean;
  onConfigChange: (instances: OllamaInstance[]) => void;
  className?: string;
}

interface ConnectionTestResult {
  isHealthy: boolean;
  responseTimeMs?: number;
  modelsAvailable?: number;
  error?: string;
}

const OllamaConfigurationPanel: React.FC<OllamaConfigurationPanelProps> = ({
  isVisible,
  onConfigChange,
  className = ''
}) => {
  const [instances, setInstances] = useState<OllamaInstance[]>([]);
  const [loading, setLoading] = useState(true);
  const [testingConnections, setTestingConnections] = useState<Set<string>>(new Set());
  const [newInstanceUrl, setNewInstanceUrl] = useState('');
  const [newInstanceName, setNewInstanceName] = useState('');
  const [showAddInstance, setShowAddInstance] = useState(false);
  const { showToast } = useToast();

  // Load instances from database
  const loadInstances = async () => {
    try {
      setLoading(true);
      
      // First try to migrate from localStorage if needed
      const migrationResult = await credentialsService.migrateOllamaFromLocalStorage();
      if (migrationResult.migrated) {
        showToast(`Migrated ${migrationResult.instanceCount} Ollama instances to database`, 'success');
      }
      
      // Load instances from database
      const databaseInstances = await credentialsService.getOllamaInstances();
      setInstances(databaseInstances);
      onConfigChange(databaseInstances);
    } catch (error) {
      console.error('Failed to load Ollama instances from database:', error);
      showToast('Failed to load Ollama configuration from database', 'error');
      
      // Fallback to localStorage
      try {
        const saved = localStorage.getItem('ollama-instances');
        if (saved) {
          const localInstances = JSON.parse(saved);
          setInstances(localInstances);
          onConfigChange(localInstances);
          showToast('Loaded Ollama configuration from local backup', 'warning');
        }
      } catch (localError) {
        console.error('Failed to load from localStorage as fallback:', localError);
      }
    } finally {
      setLoading(false);
    }
  };

  // Save instances to database
  const saveInstances = async (newInstances: OllamaInstance[]) => {
    try {
      setLoading(true);
      await credentialsService.setOllamaInstances(newInstances);
      setInstances(newInstances);
      onConfigChange(newInstances);
      
      // Also backup to localStorage for fallback
      try {
        localStorage.setItem('ollama-instances', JSON.stringify(newInstances));
      } catch (localError) {
        console.warn('Failed to backup to localStorage:', localError);
      }
    } catch (error) {
      console.error('Failed to save Ollama instances to database:', error);
      showToast('Failed to save Ollama configuration to database', 'error');
    } finally {
      setLoading(false);
    }
  };

  // Test connection to an Ollama instance
  const testConnection = async (baseUrl: string): Promise<ConnectionTestResult> => {
    try {
      const response = await fetch('/api/providers/validate', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          provider: 'ollama',
          base_url: baseUrl
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      
      return {
        isHealthy: data.health_status?.is_available || false,
        responseTimeMs: data.health_status?.response_time_ms,
        modelsAvailable: data.health_status?.models_available,
        error: data.health_status?.error_message
      };
    } catch (error) {
      return {
        isHealthy: false,
        error: error instanceof Error ? error.message : 'Unknown error'
      };
    }
  };

  // Handle connection test for a specific instance
  const handleTestConnection = async (instanceId: string) => {
    const instance = instances.find(inst => inst.id === instanceId);
    if (!instance) return;

    setTestingConnections(prev => new Set(prev).add(instanceId));

    try {
      const result = await testConnection(instance.baseUrl);
      
      // Update instance with test results
      const updatedInstances = instances.map(inst => 
        inst.id === instanceId 
          ? {
              ...inst,
              isHealthy: result.isHealthy,
              responseTimeMs: result.responseTimeMs,
              modelsAvailable: result.modelsAvailable,
              lastHealthCheck: new Date().toISOString()
            }
          : inst
      );
      saveInstances(updatedInstances);

      if (result.isHealthy) {
        showToast(`Connected to ${instance.name} (${result.responseTimeMs?.toFixed(0)}ms, ${result.modelsAvailable} models)`, 'success');
      } else {
        showToast(result.error || 'Unable to connect to Ollama instance', 'error');
      }
    } catch (error) {
      showToast(`Connection test failed: ${error instanceof Error ? error.message : 'Unknown error'}`, 'error');
    } finally {
      setTestingConnections(prev => {
        const newSet = new Set(prev);
        newSet.delete(instanceId);
        return newSet;
      });
    }
  };

  // Add new instance
  const handleAddInstance = async () => {
    if (!newInstanceUrl.trim() || !newInstanceName.trim()) {
      showToast('Please provide both URL and name for the new instance', 'error');
      return;
    }

    // Validate URL format
    try {
      const url = new URL(newInstanceUrl);
      if (!url.protocol.startsWith('http')) {
        throw new Error('URL must use HTTP or HTTPS protocol');
      }
    } catch (error) {
      showToast('Please provide a valid HTTP/HTTPS URL', 'error');
      return;
    }

    // Check for duplicate URLs
    const isDuplicate = instances.some(inst => inst.baseUrl === newInstanceUrl.trim());
    if (isDuplicate) {
      showToast('An instance with this URL already exists', 'error');
      return;
    }

    const newInstance: OllamaInstance = {
      id: `instance-${Date.now()}`,
      name: newInstanceName.trim(),
      baseUrl: newInstanceUrl.trim(),
      isEnabled: true,
      isPrimary: false,
      loadBalancingWeight: 100
    };

    try {
      setLoading(true);
      await credentialsService.addOllamaInstance(newInstance);
      
      // Reload instances from database to get updated list
      await loadInstances();
      
      setNewInstanceUrl('');
      setNewInstanceName('');
      setShowAddInstance(false);
      
      showToast(`Added new Ollama instance: ${newInstance.name}`, 'success');
    } catch (error) {
      console.error('Failed to add Ollama instance:', error);
      showToast(`Failed to add Ollama instance: ${error instanceof Error ? error.message : 'Unknown error'}`, 'error');
    } finally {
      setLoading(false);
    }
  };

  // Remove instance
  const handleRemoveInstance = async (instanceId: string) => {
    const instance = instances.find(inst => inst.id === instanceId);
    if (!instance) return;

    // Don't allow removing the last instance
    if (instances.length <= 1) {
      showToast('At least one Ollama instance must be configured', 'error');
      return;
    }

    try {
      setLoading(true);
      await credentialsService.removeOllamaInstance(instanceId);
      
      // Reload instances from database to get updated list
      await loadInstances();
      
      showToast(`Removed Ollama instance: ${instance.name}`, 'success');
    } catch (error) {
      console.error('Failed to remove Ollama instance:', error);
      showToast(`Failed to remove Ollama instance: ${error instanceof Error ? error.message : 'Unknown error'}`, 'error');
    } finally {
      setLoading(false);
    }
  };

  // Update instance URL
  const handleUpdateInstanceUrl = async (instanceId: string, newUrl: string) => {
    try {
      await credentialsService.updateOllamaInstance(instanceId, { 
        baseUrl: newUrl, 
        isHealthy: undefined, 
        lastHealthCheck: undefined 
      });
      await loadInstances(); // Reload to get updated data
    } catch (error) {
      console.error('Failed to update Ollama instance URL:', error);
      showToast('Failed to update instance URL', 'error');
    }
  };

  // Toggle instance enabled state
  const handleToggleInstance = async (instanceId: string) => {
    const instance = instances.find(inst => inst.id === instanceId);
    if (!instance) return;

    try {
      await credentialsService.updateOllamaInstance(instanceId, { 
        isEnabled: !instance.isEnabled 
      });
      await loadInstances(); // Reload to get updated data
    } catch (error) {
      console.error('Failed to toggle Ollama instance:', error);
      showToast('Failed to toggle instance state', 'error');
    }
  };

  // Set instance as primary
  const handleSetPrimary = async (instanceId: string) => {
    try {
      // Update all instances - only the specified one should be primary
      await saveInstances(instances.map(inst => ({
        ...inst,
        isPrimary: inst.id === instanceId
      })));
    } catch (error) {
      console.error('Failed to set primary Ollama instance:', error);
      showToast('Failed to set primary instance', 'error');
    }
  };

  // Load instances from database on mount
  useEffect(() => {
    loadInstances();
  }, []); // Empty dependency array - load only on mount

  // Notify parent of configuration changes
  useEffect(() => {
    onConfigChange(instances);
  }, [instances, onConfigChange]);

  // Auto-test primary instance when component becomes visible
  useEffect(() => {
    if (isVisible && instances.length > 0) {
      const primaryInstance = instances.find(inst => inst.isPrimary);
      if (primaryInstance && primaryInstance.isHealthy === undefined) {
        handleTestConnection(primaryInstance.id);
      }
    }
  }, [isVisible, instances.length]);

  if (!isVisible) return null;

  const getConnectionStatusBadge = (instance: OllamaInstance) => {
    if (testingConnections.has(instance.id)) {
      return <Badge variant="outline" className="animate-pulse">Testing...</Badge>;
    }
    
    if (instance.isHealthy === true) {
      return (
        <Badge variant="solid" className="flex items-center gap-1 bg-green-100 text-green-800 border-green-200">
          <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
          Online
          {instance.responseTimeMs && (
            <span className="text-xs opacity-75">
              ({instance.responseTimeMs.toFixed(0)}ms)
            </span>
          )}
        </Badge>
      );
    }
    
    if (instance.isHealthy === false) {
      return (
        <Badge variant="solid" className="flex items-center gap-1 bg-red-100 text-red-800 border-red-200">
          <div className="w-2 h-2 rounded-full bg-red-500" />
          Offline
        </Badge>
      );
    }
    
    return <Badge variant="outline">Unknown</Badge>;
  };

  return (
    <Card 
      accentColor="green" 
      className={cn("mt-4 space-y-4", className)}
    >
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
            Ollama Configuration
          </h3>
          <p className="text-sm text-gray-600 dark:text-gray-400">
            Configure Ollama instances for distributed processing
          </p>
        </div>
        <Badge variant="outline" className="text-xs">
          {instances.filter(inst => inst.isEnabled).length} Active
        </Badge>
      </div>

      {/* Instance List */}
      <div className="space-y-3">
        {instances.map((instance) => (
          <Card key={instance.id} className="p-4 bg-gray-50 dark:bg-gray-800/50">
            <div className="flex items-start justify-between">
              <div className="flex-1 space-y-2">
                <div className="flex items-center gap-2">
                  <span className="font-medium text-gray-900 dark:text-white">
                    {instance.name}
                  </span>
                  {instance.isPrimary && (
                    <Badge variant="outline" className="text-xs">Primary</Badge>
                  )}
                  {getConnectionStatusBadge(instance)}
                </div>
                
                <Input
                  type="url"
                  value={instance.baseUrl}
                  onChange={(e) => handleUpdateInstanceUrl(instance.id, e.target.value)}
                  placeholder="http://localhost:11434"
                  className="text-sm"
                />
                
                {instance.modelsAvailable !== undefined && (
                  <div className="text-xs text-gray-600 dark:text-gray-400">
                    {instance.modelsAvailable} models available
                  </div>
                )}
              </div>
              
              <div className="flex items-center gap-2 ml-4">
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => handleTestConnection(instance.id)}
                  disabled={testingConnections.has(instance.id)}
                  className="text-xs"
                >
                  {testingConnections.has(instance.id) ? 'Testing...' : 'Test'}
                </Button>
                
                {!instance.isPrimary && (
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => handleSetPrimary(instance.id)}
                    className="text-xs"
                  >
                    Set Primary
                  </Button>
                )}
                
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => handleToggleInstance(instance.id)}
                  className={cn(
                    "text-xs",
                    instance.isEnabled 
                      ? "text-green-600 hover:text-green-700" 
                      : "text-gray-500 hover:text-gray-600"
                  )}
                >
                  {instance.isEnabled ? 'Enabled' : 'Disabled'}
                </Button>
                
                {instances.length > 1 && (
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => handleRemoveInstance(instance.id)}
                    className="text-xs text-red-600 hover:text-red-700"
                  >
                    Remove
                  </Button>
                )}
              </div>
            </div>
          </Card>
        ))}
      </div>

      {/* Add Instance Section */}
      {showAddInstance ? (
        <Card className="p-4 bg-blue-50 dark:bg-blue-900/20 border-blue-200 dark:border-blue-800">
          <div className="space-y-3">
            <h4 className="font-medium text-blue-900 dark:text-blue-100">
              Add New Ollama Instance
            </h4>
            
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              <Input
                type="text"
                placeholder="Instance Name"
                value={newInstanceName}
                onChange={(e) => setNewInstanceName(e.target.value)}
              />
              <Input
                type="url"
                placeholder="http://localhost:11434"
                value={newInstanceUrl}
                onChange={(e) => setNewInstanceUrl(e.target.value)}
              />
            </div>
            
            <div className="flex gap-2">
              <Button
                size="sm"
                onClick={handleAddInstance}
                className="bg-blue-600 hover:bg-blue-700"
              >
                Add Instance
              </Button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => {
                  setShowAddInstance(false);
                  setNewInstanceUrl('');
                  setNewInstanceName('');
                }}
              >
                Cancel
              </Button>
            </div>
          </div>
        </Card>
      ) : (
        <Button
          variant="outline"
          onClick={() => setShowAddInstance(true)}
          className="w-full border-dashed border-2 border-gray-300 dark:border-gray-600 hover:border-gray-400 dark:hover:border-gray-500"
        >
          <span className="text-gray-600 dark:text-gray-400">+ Add Ollama Instance</span>
        </Button>
      )}

      {/* Configuration Summary */}
      <div className="pt-4 border-t border-gray-200 dark:border-gray-700">
        <div className="text-xs text-gray-600 dark:text-gray-400 space-y-1">
          <div className="flex justify-between">
            <span>Total Instances:</span>
            <span className="font-mono">{instances.length}</span>
          </div>
          <div className="flex justify-between">
            <span>Active Instances:</span>
            <span className="font-mono text-green-600 dark:text-green-400">
              {instances.filter(inst => inst.isEnabled && inst.isHealthy).length}
            </span>
          </div>
          <div className="flex justify-between">
            <span>Load Balancing:</span>
            <span className="font-mono">
              {instances.filter(inst => inst.isEnabled).length > 1 ? 'Enabled' : 'Disabled'}
            </span>
          </div>
        </div>
      </div>
    </Card>
  );
};

export default OllamaConfigurationPanel;