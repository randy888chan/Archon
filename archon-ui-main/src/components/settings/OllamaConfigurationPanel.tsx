import React, { useState, useEffect } from 'react';
import { Card } from '../ui/Card';
import { Button } from '../ui/Button';
import { Input } from '../ui/Input';
import { Badge } from '../ui/Badge';
import { useToast } from '../../contexts/ToastContext';
import { cn } from '../../lib/utils';

interface OllamaInstance {
  id: string;
  name: string;
  baseUrl: string;
  isEnabled: boolean;
  isPrimary: boolean;
  loadBalancingWeight: number;
  isHealthy?: boolean;
  responseTimeMs?: number;
  modelsAvailable?: number;
  lastHealthCheck?: string;
}

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
  const [instances, setInstances] = useState<OllamaInstance[]>([
    {
      id: 'primary',
      name: 'Primary Ollama Instance',
      baseUrl: 'http://localhost:11434',
      isEnabled: true,
      isPrimary: true,
      loadBalancingWeight: 100
    }
  ]);
  const [testingConnections, setTestingConnections] = useState<Set<string>>(new Set());
  const [newInstanceUrl, setNewInstanceUrl] = useState('');
  const [newInstanceName, setNewInstanceName] = useState('');
  const [showAddInstance, setShowAddInstance] = useState(false);
  const { showToast } = useToast();

  // Test connection to an Ollama instance
  const testConnection = async (baseUrl: string): Promise<ConnectionTestResult> => {
    try {
      const response = await fetch('/api/providers/health', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          provider: 'ollama',
          config: { base_url: baseUrl }
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
      setInstances(prev => prev.map(inst => 
        inst.id === instanceId 
          ? {
              ...inst,
              isHealthy: result.isHealthy,
              responseTimeMs: result.responseTimeMs,
              modelsAvailable: result.modelsAvailable,
              lastHealthCheck: new Date().toISOString()
            }
          : inst
      ));

      if (result.isHealthy) {
        showToast({
          title: 'Connection Successful',
          description: `Connected to ${instance.name} (${result.responseTimeMs?.toFixed(0)}ms, ${result.modelsAvailable} models)`,
          variant: 'success'
        });
      } else {
        showToast({
          title: 'Connection Failed',
          description: result.error || 'Unable to connect to Ollama instance',
          variant: 'destructive'
        });
      }
    } catch (error) {
      showToast({
        title: 'Connection Test Failed',
        description: error instanceof Error ? error.message : 'Unknown error',
        variant: 'destructive'
      });
    } finally {
      setTestingConnections(prev => {
        const newSet = new Set(prev);
        newSet.delete(instanceId);
        return newSet;
      });
    }
  };

  // Add new instance
  const handleAddInstance = () => {
    if (!newInstanceUrl.trim() || !newInstanceName.trim()) {
      showToast({
        title: 'Validation Error',
        description: 'Please provide both URL and name for the new instance',
        variant: 'destructive'
      });
      return;
    }

    // Validate URL format
    try {
      const url = new URL(newInstanceUrl);
      if (!url.protocol.startsWith('http')) {
        throw new Error('URL must use HTTP or HTTPS protocol');
      }
    } catch (error) {
      showToast({
        title: 'Invalid URL',
        description: 'Please provide a valid HTTP/HTTPS URL',
        variant: 'destructive'
      });
      return;
    }

    // Check for duplicate URLs
    const isDuplicate = instances.some(inst => inst.baseUrl === newInstanceUrl.trim());
    if (isDuplicate) {
      showToast({
        title: 'Duplicate Instance',
        description: 'An instance with this URL already exists',
        variant: 'destructive'
      });
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

    setInstances(prev => [...prev, newInstance]);
    setNewInstanceUrl('');
    setNewInstanceName('');
    setShowAddInstance(false);
    
    showToast({
      title: 'Instance Added',
      description: `Added new Ollama instance: ${newInstance.name}`,
      variant: 'success'
    });
  };

  // Remove instance
  const handleRemoveInstance = (instanceId: string) => {
    const instance = instances.find(inst => inst.id === instanceId);
    if (!instance) return;

    // Don't allow removing the last instance
    if (instances.length <= 1) {
      showToast({
        title: 'Cannot Remove',
        description: 'At least one Ollama instance must be configured',
        variant: 'destructive'
      });
      return;
    }

    setInstances(prev => {
      const filtered = prev.filter(inst => inst.id !== instanceId);
      
      // If we're removing the primary instance, make the first remaining one primary
      if (instance.isPrimary && filtered.length > 0) {
        filtered[0] = { ...filtered[0], isPrimary: true };
      }
      
      return filtered;
    });

    showToast({
      title: 'Instance Removed',
      description: `Removed Ollama instance: ${instance.name}`,
      variant: 'success'
    });
  };

  // Update instance URL
  const handleUpdateInstanceUrl = (instanceId: string, newUrl: string) => {
    setInstances(prev => prev.map(inst =>
      inst.id === instanceId 
        ? { ...inst, baseUrl: newUrl, isHealthy: undefined, lastHealthCheck: undefined }
        : inst
    ));
  };

  // Toggle instance enabled state
  const handleToggleInstance = (instanceId: string) => {
    setInstances(prev => prev.map(inst =>
      inst.id === instanceId 
        ? { ...inst, isEnabled: !inst.isEnabled }
        : inst
    ));
  };

  // Set instance as primary
  const handleSetPrimary = (instanceId: string) => {
    setInstances(prev => prev.map(inst => ({
      ...inst,
      isPrimary: inst.id === instanceId
    })));
  };

  // Notify parent of configuration changes
  useEffect(() => {
    onConfigChange(instances);
  }, [instances, onConfigChange]);

  // Auto-test primary instance on mount
  useEffect(() => {
    if (isVisible && instances.length > 0) {
      const primaryInstance = instances.find(inst => inst.isPrimary);
      if (primaryInstance && primaryInstance.isHealthy === undefined) {
        handleTestConnection(primaryInstance.id);
      }
    }
  }, [isVisible]);

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