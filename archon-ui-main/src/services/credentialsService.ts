export interface Credential {
  id?: string;
  key: string;
  value?: string;
  encrypted_value?: string;
  is_encrypted: boolean;
  category: string;
  description?: string;
  created_at?: string;
  updated_at?: string;
}

export interface RagSettings {
  USE_CONTEXTUAL_EMBEDDINGS: boolean;
  CONTEXTUAL_EMBEDDINGS_MAX_WORKERS: number;
  USE_HYBRID_SEARCH: boolean;
  USE_AGENTIC_RAG: boolean;
  USE_RERANKING: boolean;
  MODEL_CHOICE: string;
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
}

export interface CodeExtractionSettings {
  MIN_CODE_BLOCK_LENGTH: number;
  MAX_CODE_BLOCK_LENGTH: number;
  ENABLE_COMPLETE_BLOCK_DETECTION: boolean;
  ENABLE_LANGUAGE_SPECIFIC_PATTERNS: boolean;
  ENABLE_PROSE_FILTERING: boolean;
  MAX_PROSE_RATIO: number;
  MIN_CODE_INDICATORS: number;
  ENABLE_DIAGRAM_FILTERING: boolean;
  ENABLE_CONTEXTUAL_LENGTH: boolean;
  CODE_EXTRACTION_MAX_WORKERS: number;
  CONTEXT_WINDOW_SIZE: number;
  ENABLE_CODE_SUMMARIES: boolean;
}

export interface OllamaInstance {
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

import { getApiUrl } from '../config/api';

class CredentialsService {
  private baseUrl = getApiUrl();

  async getAllCredentials(): Promise<Credential[]> {
    const response = await fetch(`${this.baseUrl}/api/credentials`);
    if (!response.ok) {
      throw new Error('Failed to fetch credentials');
    }
    return response.json();
  }

  async getCredentialsByCategory(category: string): Promise<Credential[]> {
    const response = await fetch(`${this.baseUrl}/api/credentials/categories/${category}`);
    if (!response.ok) {
      throw new Error(`Failed to fetch credentials for category: ${category}`);
    }
    const result = await response.json();
    
    // The API returns {credentials: {...}} where credentials is a dict
    // Convert to array format expected by frontend
    if (result.credentials && typeof result.credentials === 'object') {
      return Object.entries(result.credentials).map(([key, value]: [string, any]) => {
        if (value && typeof value === 'object' && value.is_encrypted) {
          return {
            key,
            value: undefined,
            encrypted_value: value.encrypted_value,
            is_encrypted: true,
            category,
            description: value.description
          };
        } else {
          return {
            key,
            value: value,
            encrypted_value: undefined,
            is_encrypted: false,
            category,
            description: ''
          };
        }
      });
    }
    
    return [];
  }

  async getCredential(key: string): Promise<{ key: string; value?: string; is_encrypted?: boolean }> {
    const response = await fetch(`${this.baseUrl}/api/credentials/${key}`);
    if (!response.ok) {
      if (response.status === 404) {
        // Return empty object if credential not found
        return { key, value: undefined };
      }
      throw new Error(`Failed to fetch credential: ${key}`);
    }
    return response.json();
  }

  async getRagSettings(): Promise<RagSettings> {
    const ragCredentials = await this.getCredentialsByCategory('rag_strategy');
    const apiKeysCredentials = await this.getCredentialsByCategory('api_keys');
    
    const settings: RagSettings = {
      USE_CONTEXTUAL_EMBEDDINGS: false,
      CONTEXTUAL_EMBEDDINGS_MAX_WORKERS: 3,
      USE_HYBRID_SEARCH: true,
      USE_AGENTIC_RAG: true,
      USE_RERANKING: true,
      MODEL_CHOICE: 'gpt-4.1-nano',
      LLM_PROVIDER: 'openai',
      LLM_BASE_URL: '',
      EMBEDDING_MODEL: '',
      // Crawling Performance Settings defaults
      CRAWL_BATCH_SIZE: 50,
      CRAWL_MAX_CONCURRENT: 10,
      CRAWL_WAIT_STRATEGY: 'domcontentloaded',
      CRAWL_PAGE_TIMEOUT: 60000,  // Increased from 30s to 60s for documentation sites
      CRAWL_DELAY_BEFORE_HTML: 0.5,
      // Storage Performance Settings defaults
      DOCUMENT_STORAGE_BATCH_SIZE: 50,
      EMBEDDING_BATCH_SIZE: 100,
      DELETE_BATCH_SIZE: 100,
      ENABLE_PARALLEL_BATCHES: true,
      // Advanced Settings defaults
      MEMORY_THRESHOLD_PERCENT: 80,
      DISPATCHER_CHECK_INTERVAL: 30,
      CODE_EXTRACTION_BATCH_SIZE: 50,
      CODE_SUMMARY_MAX_WORKERS: 3
    };

    // Map credentials to settings
    [...ragCredentials, ...apiKeysCredentials].forEach(cred => {
      if (cred.key in settings) {
        // String fields
        if (['MODEL_CHOICE', 'LLM_PROVIDER', 'LLM_BASE_URL', 'EMBEDDING_MODEL', 'CRAWL_WAIT_STRATEGY'].includes(cred.key)) {
          (settings as any)[cred.key] = cred.value || '';
        } 
        // Number fields
        else if (['CONTEXTUAL_EMBEDDINGS_MAX_WORKERS', 'CRAWL_BATCH_SIZE', 'CRAWL_MAX_CONCURRENT', 
                  'CRAWL_PAGE_TIMEOUT', 'DOCUMENT_STORAGE_BATCH_SIZE', 'EMBEDDING_BATCH_SIZE', 
                  'DELETE_BATCH_SIZE', 'MEMORY_THRESHOLD_PERCENT', 'DISPATCHER_CHECK_INTERVAL',
                  'CODE_EXTRACTION_BATCH_SIZE', 'CODE_SUMMARY_MAX_WORKERS'].includes(cred.key)) {
          (settings as any)[cred.key] = parseInt(cred.value || '0', 10) || (settings as any)[cred.key];
        }
        // Float fields
        else if (cred.key === 'CRAWL_DELAY_BEFORE_HTML') {
          settings[cred.key] = parseFloat(cred.value || '0.5') || 0.5;
        }
        // Boolean fields
        else {
          (settings as any)[cred.key] = cred.value === 'true';
        }
      }
    });

    return settings;
  }

  async updateCredential(credential: Credential): Promise<Credential> {
    const response = await fetch(`${this.baseUrl}/api/credentials/${credential.key}`, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(credential),
    });
    
    if (!response.ok) {
      throw new Error('Failed to update credential');
    }
    
    return response.json();
  }

  async createCredential(credential: Credential): Promise<Credential> {
    const response = await fetch(`${this.baseUrl}/api/credentials`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(credential),
    });
    
    if (!response.ok) {
      throw new Error('Failed to create credential');
    }
    
    return response.json();
  }

  async deleteCredential(key: string): Promise<void> {
    const response = await fetch(`${this.baseUrl}/api/credentials/${key}`, {
      method: 'DELETE',
    });
    
    if (!response.ok) {
      throw new Error('Failed to delete credential');
    }
  }

  async updateRagSettings(settings: RagSettings): Promise<void> {
    const promises = [];
    
    // Update all RAG strategy settings
    for (const [key, value] of Object.entries(settings)) {
      // Skip undefined values
      if (value === undefined) continue;
      
      promises.push(
        this.updateCredential({
          key,
          value: value.toString(),
          is_encrypted: false,
          category: 'rag_strategy',
        })
      );
    }
    
    await Promise.all(promises);
  }

  async getCodeExtractionSettings(): Promise<CodeExtractionSettings> {
    const codeExtractionCredentials = await this.getCredentialsByCategory('code_extraction');
    
    const settings: CodeExtractionSettings = {
      MIN_CODE_BLOCK_LENGTH: 250,
      MAX_CODE_BLOCK_LENGTH: 5000,
      ENABLE_COMPLETE_BLOCK_DETECTION: true,
      ENABLE_LANGUAGE_SPECIFIC_PATTERNS: true,
      ENABLE_PROSE_FILTERING: true,
      MAX_PROSE_RATIO: 0.15,
      MIN_CODE_INDICATORS: 3,
      ENABLE_DIAGRAM_FILTERING: true,
      ENABLE_CONTEXTUAL_LENGTH: true,
      CODE_EXTRACTION_MAX_WORKERS: 3,
      CONTEXT_WINDOW_SIZE: 1000,
      ENABLE_CODE_SUMMARIES: true
    };

    // Map credentials to settings
    codeExtractionCredentials.forEach(cred => {
      if (cred.key in settings) {
        const key = cred.key as keyof CodeExtractionSettings;
        if (typeof settings[key] === 'number') {
          if (key === 'MAX_PROSE_RATIO') {
            settings[key] = parseFloat(cred.value || '0.15');
          } else {
            settings[key] = parseInt(cred.value || settings[key].toString(), 10);
          }
        } else if (typeof settings[key] === 'boolean') {
          settings[key] = cred.value === 'true';
        }
      }
    });

    return settings;
  }

  async updateCodeExtractionSettings(settings: CodeExtractionSettings): Promise<void> {
    const promises = [];
    
    // Update all code extraction settings
    for (const [key, value] of Object.entries(settings)) {
      promises.push(
        this.updateCredential({
          key,
          value: value.toString(),
          is_encrypted: false,
          category: 'code_extraction',
        })
      );
    }
    
    await Promise.all(promises);
  }

  // Ollama Instance Management Methods
  async getOllamaInstances(): Promise<OllamaInstance[]> {
    try {
      const response = await fetch(`${this.baseUrl}/api/provider-config/current`);
      if (!response.ok) {
        throw new Error('Failed to fetch Ollama instances from database');
      }
      
      const data = await response.json();
      
      // Convert API format to frontend format
      const instances: OllamaInstance[] = data.ollama_instances?.map((inst: any) => ({
        id: inst.id,
        name: inst.name,
        baseUrl: inst.base_url,
        isEnabled: inst.is_enabled,
        isPrimary: inst.is_primary,
        loadBalancingWeight: inst.load_balancing_weight,
        isHealthy: inst.is_healthy,
        responseTimeMs: inst.response_time_ms,
        modelsAvailable: inst.models_available,
        lastHealthCheck: inst.last_health_check
      })) || [];
      
      return instances;
    } catch (error) {
      console.error('Error fetching Ollama instances from database:', error);
      throw error;
    }
  }

  async setOllamaInstances(instances: OllamaInstance[]): Promise<void> {
    try {
      // Convert frontend format to API format
      const apiInstances = instances.map(inst => ({
        id: inst.id,
        name: inst.name,
        base_url: inst.baseUrl,
        is_enabled: inst.isEnabled,
        is_primary: inst.isPrimary,
        load_balancing_weight: inst.loadBalancingWeight,
        health_check_enabled: true
      }));

      const response = await fetch(`${this.baseUrl}/api/provider-config/update`, {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          llm_provider: 'ollama', // Assuming ollama provider
          embedding_provider: 'ollama',
          ollama_instances: apiInstances,
          provider_preferences: {}
        }),
      });

      if (!response.ok) {
        const errorData = await response.text();
        throw new Error(`Failed to save Ollama instances to database: ${errorData}`);
      }
    } catch (error) {
      console.error('Error saving Ollama instances to database:', error);
      throw error;
    }
  }

  async addOllamaInstance(instance: OllamaInstance): Promise<void> {
    try {
      // Convert frontend format to API format
      const apiInstance = {
        id: instance.id,
        name: instance.name,
        base_url: instance.baseUrl,
        is_enabled: instance.isEnabled,
        is_primary: instance.isPrimary,
        load_balancing_weight: instance.loadBalancingWeight,
        health_check_enabled: true
      };

      const response = await fetch(`${this.baseUrl}/api/provider-config/ollama/add-instance`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(apiInstance),
      });

      if (!response.ok) {
        const errorData = await response.text();
        throw new Error(`Failed to add Ollama instance to database: ${errorData}`);
      }
    } catch (error) {
      console.error('Error adding Ollama instance to database:', error);
      throw error;
    }
  }

  async removeOllamaInstance(instanceId: string): Promise<void> {
    try {
      const response = await fetch(`${this.baseUrl}/api/provider-config/ollama/remove-instance/${instanceId}`, {
        method: 'DELETE',
      });

      if (!response.ok) {
        const errorData = await response.text();
        throw new Error(`Failed to remove Ollama instance from database: ${errorData}`);
      }
    } catch (error) {
      console.error('Error removing Ollama instance from database:', error);
      throw error;
    }
  }

  async updateOllamaInstance(instanceId: string, updates: Partial<OllamaInstance>): Promise<void> {
    try {
      // Get current instances, update the specific one, then save all
      const instances = await this.getOllamaInstances();
      const instanceIndex = instances.findIndex(inst => inst.id === instanceId);
      
      if (instanceIndex === -1) {
        throw new Error(`Ollama instance with ID ${instanceId} not found`);
      }

      // Apply updates
      instances[instanceIndex] = { ...instances[instanceIndex], ...updates };

      // Save updated instances
      await this.setOllamaInstances(instances);
    } catch (error) {
      console.error('Error updating Ollama instance in database:', error);
      throw error;
    }
  }

  async migrateOllamaFromLocalStorage(): Promise<{ migrated: boolean; instanceCount: number }> {
    try {
      // Check if localStorage has Ollama instances
      const localStorageData = localStorage.getItem('ollama-instances');
      if (!localStorageData) {
        return { migrated: false, instanceCount: 0 };
      }

      const localInstances = JSON.parse(localStorageData);
      if (!Array.isArray(localInstances) || localInstances.length === 0) {
        return { migrated: false, instanceCount: 0 };
      }

      // Check if database already has instances
      const existingInstances = await this.getOllamaInstances();
      if (existingInstances.length > 0) {
        // Database already has instances, don't migrate
        return { migrated: false, instanceCount: existingInstances.length };
      }

      // Migrate localStorage instances to database
      const instancesToMigrate: OllamaInstance[] = localInstances.map((inst: any) => ({
        id: inst.id || `migrated-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
        name: inst.name || 'Migrated Instance',
        baseUrl: inst.baseUrl || inst.url || 'http://localhost:11434',
        isEnabled: inst.isEnabled !== false, // Default to true
        isPrimary: inst.isPrimary || false,
        loadBalancingWeight: inst.loadBalancingWeight || 100,
        isHealthy: inst.isHealthy,
        responseTimeMs: inst.responseTimeMs,
        modelsAvailable: inst.modelsAvailable,
        lastHealthCheck: inst.lastHealthCheck
      }));

      // Ensure at least one instance is marked as primary
      if (!instancesToMigrate.some(inst => inst.isPrimary)) {
        instancesToMigrate[0].isPrimary = true;
      }

      await this.setOllamaInstances(instancesToMigrate);

      console.log(`Successfully migrated ${instancesToMigrate.length} Ollama instances from localStorage to database`);
      
      return { migrated: true, instanceCount: instancesToMigrate.length };
    } catch (error) {
      console.error('Error migrating Ollama instances from localStorage:', error);
      throw error;
    }
  }
}

export const credentialsService = new CredentialsService(); 