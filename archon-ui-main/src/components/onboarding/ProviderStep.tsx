import { useState } from 'react';
import { Key, ExternalLink, Save, Loader } from 'lucide-react';
import { Input } from '../ui/Input';
import { Button } from '../ui/Button';
import { Select } from '../ui/Select';
import { useToast } from '../../contexts/ToastContext';
import { credentialsService } from '../../services/credentialsService';

interface ProviderStepProps {
  onSaved: () => void;
  onSkip: () => void;
}

export const ProviderStep = ({ onSaved, onSkip }: ProviderStepProps) => {
  const [provider, setProvider] = useState('openai');
  const [apiKey, setApiKey] = useState('');
  const [azureEndpoint, setAzureEndpoint] = useState('');
  const [azureApiVersion, setAzureApiVersion] = useState('2024-12-01-preview');
  const [saving, setSaving] = useState(false);
  const { showToast } = useToast();

  const handleSave = async () => {
    if (!apiKey.trim()) {
      showToast('Please enter an API key', 'error');
      return;
    }

    if (provider === 'azure' && !azureEndpoint.trim()) {
      showToast('Please enter your Azure OpenAI endpoint', 'error');
      return;
    }

    setSaving(true);
    try {
      // Save the API key based on provider
      const apiKeyName = provider === 'azure' ? 'AZURE_OPENAI_API_KEY' : 'OPENAI_API_KEY';
      await credentialsService.createCredential({
        key: apiKeyName,
        value: apiKey,
        is_encrypted: true,
        category: 'api_keys'
      });

      // Save Azure-specific settings if Azure is selected
      if (provider === 'azure') {
        await credentialsService.updateCredential({
          key: 'AZURE_OPENAI_ENDPOINT',
          value: azureEndpoint,
          is_encrypted: false,
          category: 'rag_strategy'
        });

        await credentialsService.updateCredential({
          key: 'AZURE_API_VERSION',
          value: azureApiVersion,
          is_encrypted: false,
          category: 'rag_strategy'
        });
      }

      // Update the provider setting
      await credentialsService.updateCredential({
        key: 'LLM_PROVIDER',
        value: provider,
        is_encrypted: false,
        category: 'rag_strategy'
      });

      showToast('API key saved successfully!', 'success');
      // Mark onboarding as dismissed when API key is saved
      localStorage.setItem('onboardingDismissed', 'true');
      onSaved();
    } catch (error) {
      // Detailed error handling for critical configuration per alpha principles
      const errorMessage = error instanceof Error ? error.message : 'Unknown error';
      const errorDetails = {
        context: 'API key configuration',
        operation: `save_${provider}_key`,
        provider: provider,
        error: errorMessage,
        timestamp: new Date().toISOString()
      };
      
      // Log with full context and stack trace
      console.error('API_KEY_SAVE_FAILED:', errorDetails, error);
      
      // Show specific error details to help user resolve the issue
      if (errorMessage.includes('duplicate') || errorMessage.includes('already exists')) {
        showToast(
          'API key already exists. Please update it in Settings if you want to change it.',
          'warning'
        );
      } else if (errorMessage.includes('network') || errorMessage.includes('fetch')) {
        showToast(
          `Network error while saving API key: ${errorMessage}. Please check your connection.`,
          'error'
        );
      } else if (errorMessage.includes('unauthorized') || errorMessage.includes('forbidden')) {
        showToast(
          `Permission error: ${errorMessage}. Please check backend configuration.`,
          'error'
        );
      } else {
        // Show the actual error for unknown issues
        showToast(
          `Failed to save API key: ${errorMessage}`,
          'error'
        );
      }
    } finally {
      setSaving(false);
    }
  };

  const handleSkip = () => {
    showToast('You can configure your provider in Settings', 'info');
    // Mark onboarding as dismissed when skipping
    localStorage.setItem('onboardingDismissed', 'true');
    onSkip();
  };

  return (
    <div className="space-y-6">
      {/* Provider Selection */}
      <div>
        <Select
          label="Select AI Provider"
          value={provider}
          onChange={(e) => setProvider(e.target.value)}
          options={[
            { value: 'openai', label: 'OpenAI' },
            { value: 'azure', label: 'Azure OpenAI' },
            { value: 'google', label: 'Google Gemini' },
            { value: 'ollama', label: 'Ollama (Local)' },
          ]}
          accentColor="green"
        />
        <p className="mt-2 text-sm text-gray-600 dark:text-zinc-400">
          {provider === 'openai' && 'OpenAI provides powerful models like GPT-4. You\'ll need an API key from OpenAI.'}
          {provider === 'azure' && 'Azure OpenAI provides enterprise-grade AI with your own Azure subscription and deployment.'}
          {provider === 'google' && 'Google Gemini offers advanced AI capabilities. Configure in Settings after setup.'}
          {provider === 'ollama' && 'Ollama runs models locally on your machine. Configure in Settings after setup.'}
        </p>
      </div>

      {/* OpenAI API Key Input */}
      {provider === 'openai' && (
        <>
          <div>
            <Input
              label="OpenAI API Key"
              type="password"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder="sk-..."
              accentColor="green"
              icon={<Key className="w-4 h-4" />}
            />
            <p className="mt-2 text-sm text-gray-600 dark:text-zinc-400">
              Your API key will be encrypted and stored securely.
            </p>
          </div>

          <div className="flex items-center gap-2 text-sm">
            <a
              href="https://platform.openai.com/api-keys"
              target="_blank"
              rel="noopener noreferrer"
              className="text-blue-500 hover:text-blue-600 dark:text-blue-400 dark:hover:text-blue-300 flex items-center gap-1"
            >
              Get an API key from OpenAI
              <ExternalLink className="w-3 h-3" />
            </a>
          </div>

          <div className="flex gap-3 pt-4">
            <Button
              variant="primary"
              size="lg"
              onClick={handleSave}
              disabled={saving || !apiKey.trim()}
              icon={saving ? <Loader className="w-4 h-4 animate-spin" /> : <Save className="w-4 h-4" />}
              className="flex-1"
            >
              {saving ? 'Saving...' : 'Save & Continue'}
            </Button>
            <Button
              variant="outline"
              size="lg"
              onClick={handleSkip}
              disabled={saving}
              className="flex-1"
            >
              Skip for Now
            </Button>
          </div>
        </>
      )}

      {/* Azure OpenAI Configuration */}
      {provider === 'azure' && (
        <>
          <div>
            <Input
              label="Azure OpenAI API Key"
              type="password"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder="your-azure-openai-api-key"
              accentColor="green"
              icon={<Key className="w-4 h-4" />}
            />
            <p className="mt-2 text-sm text-gray-600 dark:text-zinc-400">
              Your API key will be encrypted and stored securely.
            </p>
          </div>

          <div>
            <Input
              label="Azure OpenAI Endpoint"
              type="text"
              value={azureEndpoint}
              onChange={(e) => setAzureEndpoint(e.target.value)}
              placeholder="https://your-resource.openai.azure.com"
              accentColor="green"
            />
            <p className="mt-2 text-sm text-gray-600 dark:text-zinc-400">
              Your Azure OpenAI resource endpoint URL.
            </p>
          </div>

          <div>
            <Input
              label="API Version"
              type="text"
              value={azureApiVersion}
              onChange={(e) => setAzureApiVersion(e.target.value)}
              placeholder="2024-12-01-preview"
              accentColor="green"
            />
            <p className="mt-2 text-sm text-gray-600 dark:text-zinc-400">
              Azure OpenAI API version (leave default if unsure).
            </p>
          </div>

          <div className="flex items-center gap-2 text-sm">
            <a
              href="https://portal.azure.com/#view/Microsoft_Azure_ProjectOxford/CognitiveServicesHub/~/OpenAI"
              target="_blank"
              rel="noopener noreferrer"
              className="text-blue-500 hover:text-blue-600 dark:text-blue-400 dark:hover:text-blue-300 flex items-center gap-1"
            >
              Create Azure OpenAI resource
              <ExternalLink className="w-3 h-3" />
            </a>
          </div>

          <div className="flex gap-3 pt-4">
            <Button
              variant="primary"
              size="lg"
              onClick={handleSave}
              disabled={saving || !apiKey.trim() || !azureEndpoint.trim()}
              icon={saving ? <Loader className="w-4 h-4 animate-spin" /> : <Save className="w-4 h-4" />}
              className="flex-1"
            >
              {saving ? 'Saving...' : 'Save & Continue'}
            </Button>
            <Button
              variant="outline"
              size="lg"
              onClick={handleSkip}
              disabled={saving}
              className="flex-1"
            >
              Skip for Now
            </Button>
          </div>
        </>
      )}

      {/* Non-OpenAI/Azure Provider Message */}
      {provider !== 'openai' && provider !== 'azure' && (
        <div className="space-y-4">
          <div className="p-4 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg">
            <p className="text-sm text-blue-800 dark:text-blue-200">
              {provider === 'google' && 'Google Gemini configuration will be available in Settings after setup.'}
              {provider === 'ollama' && 'Ollama configuration will be available in Settings after setup. Make sure Ollama is running locally.'}
            </p>
          </div>
          
          <div className="flex gap-3 pt-4">
            <Button
              variant="primary"
              size="lg"
              onClick={async () => {
                // Save the provider selection for non-OpenAI providers
                try {
                  await credentialsService.updateCredential({
                    key: 'LLM_PROVIDER',
                    value: provider,
                    is_encrypted: false,
                    category: 'rag_strategy'
                  });
                  showToast(`${provider === 'google' ? 'Google Gemini' : 'Ollama'} selected as provider`, 'success');
                  // Mark onboarding as dismissed
                  localStorage.setItem('onboardingDismissed', 'true');
                  onSaved();
                } catch (error) {
                  console.error('Failed to save provider selection:', error);
                  showToast('Failed to save provider selection', 'error');
                }
              }}
              className="flex-1"
            >
              Continue with {provider === 'google' ? 'Gemini' : 'Ollama'}
            </Button>
          </div>
        </div>
      )}
    </div>
  );
};