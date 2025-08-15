import React from 'react';
import { motion } from 'framer-motion';
import {
  CheckCircle2,
  AlertCircle,
  Loader2,
  Zap,
  Eye,
  Layers,
  Activity,
  DollarSign,
  Database,
  Clock,
  Users,
  Cpu,
  CheckCircle,
  Download,
  Play,
  RefreshCw
} from 'lucide-react';
import { Badge } from '../ui/Badge';
import { Button } from '../ui/Button';
import { ModelSpec, Provider } from './ModelSelectionModal';

interface ModelSpecificationCardProps {
  model: ModelSpec;
  isSelected: boolean;
  onSelect: (model: ModelSpec) => void;
  loading?: boolean;
  className?: string;
}

interface CapabilityBadgeProps {
  capability: string;
  type?: 'primary' | 'secondary' | 'success' | 'warning';
}

const CapabilityBadge: React.FC<CapabilityBadgeProps> = ({ capability, type = 'secondary' }) => {
  const getIconForCapability = (cap: string) => {
    const lowerCap = cap.toLowerCase();
    if (lowerCap.includes('vision') || lowerCap.includes('image')) return <Eye className="w-3 h-3" />;
    if (lowerCap.includes('function') || lowerCap.includes('tool')) return <Zap className="w-3 h-3" />;
    if (lowerCap.includes('code')) return <Cpu className="w-3 h-3" />;
    if (lowerCap.includes('text')) return <Database className="w-3 h-3" />;
    if (lowerCap.includes('embedding')) return <Layers className="w-3 h-3" />;
    return null;
  };

  const colorMap = {
    primary: 'blue',
    secondary: 'gray', 
    success: 'green',
    warning: 'orange'
  } as const;

  const icon = getIconForCapability(capability);

  return (
    <Badge color={colorMap[type]} variant="outline" className="text-xs flex items-center gap-1">
      {icon}
      <span>{capability}</span>
    </Badge>
  );
};

const StatusIndicator: React.FC<{ status: ModelSpec['status'] }> = ({ status }) => {
  const statusConfig = {
    available: {
      icon: <CheckCircle2 className="w-4 h-4 text-green-400" />,
      label: 'Available',
      color: 'text-green-400'
    },
    installing: {
      icon: <Loader2 className="w-4 h-4 text-blue-400 animate-spin" />,
      label: 'Installing',
      color: 'text-blue-400'
    },
    error: {
      icon: <AlertCircle className="w-4 h-4 text-red-400" />,
      label: 'Error',
      color: 'text-red-400'
    },
    unavailable: {
      icon: <AlertCircle className="w-4 h-4 text-gray-400" />,
      label: 'Unavailable',
      color: 'text-gray-400'
    }
  };

  const config = statusConfig[status];
  
  return (
    <div className="flex items-center gap-2">
      {config.icon}
      <span className={`text-xs font-medium ${config.color}`}>
        {config.label}
      </span>
    </div>
  );
};

const PerformanceIndicator: React.FC<{ 
  speed: ModelSpec['performance']['speed']; 
  quality: ModelSpec['performance']['quality'] 
}> = ({ speed, quality }) => {
  const getSpeedColor = (speed: string) => {
    switch (speed) {
      case 'fast': return 'green';
      case 'medium': return 'orange'; 
      case 'slow': return 'gray';
      default: return 'gray';
    }
  };

  const getQualityColor = (quality: string) => {
    switch (quality) {
      case 'high': return 'green';
      case 'medium': return 'orange';
      case 'low': return 'gray';
      default: return 'gray';
    }
  };

  return (
    <div className="flex items-center gap-2">
      <Activity className="w-3 h-3 text-gray-500" />
      <div className="flex gap-1">
        <Badge color={getSpeedColor(speed)} variant="outline" className="text-xs">
          {speed}
        </Badge>
        <Badge color={getQualityColor(quality)} variant="outline" className="text-xs">
          {quality} quality
        </Badge>
      </div>
    </div>
  );
};

const PricingDisplay: React.FC<{ pricing?: ModelSpec['pricing'] }> = ({ pricing }) => {
  if (!pricing) return null;
  
  return (
    <div className="flex items-center gap-2 text-xs">
      <DollarSign className="w-3 h-3 text-gray-500" />
      <span className="text-gray-400">Cost:</span>
      <div className="flex gap-1">
        {pricing.input && (
          <span className="text-gray-200">
            ${pricing.input.toFixed(pricing.input < 0.001 ? 5 : 3)}/1K in
          </span>
        )}
        {pricing.output && (
          <span className="text-gray-200">
            ${pricing.output.toFixed(pricing.output < 0.001 ? 5 : 3)}/1K out
          </span>
        )}
      </div>
    </div>
  );
};

const getProviderColorClass = (provider: Provider): string => {
  switch (provider) {
    case 'openai': return 'green';
    case 'google': return 'blue';
    case 'ollama': return 'orange';
    case 'anthropic': return 'purple';
    default: return 'gray';
  }
};

const formatContextWindow = (tokens: number): string => {
  if (tokens >= 1000000) {
    return `${(tokens / 1000000).toFixed(1)}M`;
  } else if (tokens >= 1000) {
    return `${(tokens / 1000).toFixed(0)}K`;
  }
  return tokens.toString();
};

export const ModelSpecificationCard: React.FC<ModelSpecificationCardProps> = ({
  model,
  isSelected,
  onSelect,
  loading = false,
  className = ''
}) => {
  const isDisabled = model.status !== 'available' || loading;
  const providerColor = getProviderColorClass(model.provider);
  
  const handleClick = () => {
    if (!isDisabled) {
      onSelect(model);
    }
  };

  const getActionButton = () => {
    if (loading) {
      return (
        <Button
          variant="outline"
          size="sm"
          className="w-full"
          disabled
        >
          <Loader2 className="w-4 h-4 mr-2 animate-spin" />
          Loading...
        </Button>
      );
    }

    switch (model.status) {
      case 'available':
        return (
          <Button
            variant={isSelected ? "primary" : "outline"}
            accentColor={providerColor}
            size="sm"
            className="w-full"
            onClick={(e) => {
              e.stopPropagation();
              handleClick();
            }}
          >
            {isSelected ? (
              <>
                <CheckCircle className="w-4 h-4 mr-2" />
                Selected
              </>
            ) : (
              <>
                <Play className="w-4 h-4 mr-2" />
                Select Model
              </>
            )}
          </Button>
        );
      case 'installing':
        return (
          <Button
            variant="outline"
            size="sm"
            className="w-full"
            disabled
          >
            <Download className="w-4 h-4 mr-2" />
            Installing...
          </Button>
        );
      case 'error':
        return (
          <Button
            variant="outline"
            size="sm"
            className="w-full"
            onClick={(e) => {
              e.stopPropagation();
              // Could trigger retry logic here
            }}
          >
            <RefreshCw className="w-4 h-4 mr-2" />
            Retry
          </Button>
        );
      case 'unavailable':
      default:
        return (
          <Button
            variant="outline"
            size="sm"
            className="w-full"
            disabled
          >
            <Download className="w-4 h-4 mr-2" />
            Install
          </Button>
        );
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className={`
        relative p-4 rounded-lg border-2 transition-all duration-300 cursor-pointer
        bg-gradient-to-b from-white/5 to-black/20 backdrop-blur-sm
        hover:shadow-lg hover:scale-[1.02]
        ${isSelected && !isDisabled
          ? `border-${providerColor}-500 bg-${providerColor}-500/10 shadow-[0_0_20px_rgba(59,130,246,0.3)]`
          : model.status === 'available'
          ? 'border-gray-600 hover:border-gray-500'
          : 'border-gray-700 opacity-75'
        }
        ${isDisabled ? 'cursor-not-allowed' : ''}
        ${className}
      `}
      onClick={handleClick}
    >
      {/* Recommended badge */}
      {model.recommended && (
        <div className="absolute top-2 right-2">
          <Badge color={providerColor} variant="solid" className="text-xs">
            Recommended
          </Badge>
        </div>
      )}

      {/* Status and type indicators */}
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <StatusIndicator status={model.status} />
          <Badge color={providerColor} variant="outline" className="text-xs capitalize">
            {model.type}
          </Badge>
        </div>
        {model.toolSupport && (
          <div className="flex items-center gap-1 text-xs text-green-400">
            <Zap className="w-3 h-3" />
            <span>Tools</span>
          </div>
        )}
      </div>

      {/* Model name and description */}
      <h3 className={`font-semibold text-lg mb-2 ${
        isSelected ? `text-${providerColor}-400` : 'text-gray-200'
      }`}>
        {model.displayName}
      </h3>
      
      <p className="text-xs text-gray-400 leading-tight mb-3 line-clamp-2">
        {model.description}
      </p>

      {/* Model specifications */}
      <div className="space-y-2 mb-3">
        {/* Context Window */}
        <div className="flex items-center gap-2 text-xs">
          <Database className="w-3 h-3 text-gray-500" />
          <span className="text-gray-400">Context:</span>
          <span className="text-gray-200 font-medium">
            {formatContextWindow(model.contextWindow)} tokens
          </span>
        </div>

        {/* Dimensions for embedding models */}
        {model.type === 'embedding' && model.dimensions && (
          <div className="flex items-center gap-2 text-xs">
            <Layers className="w-3 h-3 text-gray-500" />
            <span className="text-gray-400">Dimensions:</span>
            <Badge color={providerColor} variant="outline" className="text-xs">
              {model.dimensions}d
            </Badge>
          </div>
        )}

        {/* Max Tokens for chat models */}
        {model.type === 'chat' && model.maxTokens && (
          <div className="flex items-center gap-2 text-xs">
            <Cpu className="w-3 h-3 text-gray-500" />
            <span className="text-gray-400">Max Output:</span>
            <span className="text-gray-200">
              {formatContextWindow(model.maxTokens)} tokens
            </span>
          </div>
        )}

        {/* Performance indicators */}
        <PerformanceIndicator 
          speed={model.performance.speed} 
          quality={model.performance.quality} 
        />

        {/* Pricing information */}
        <PricingDisplay pricing={model.pricing} />
      </div>

      {/* Capabilities */}
      {model.capabilities.length > 0 && (
        <div className="mb-3">
          <div className="flex flex-wrap gap-1 mb-2">
            {model.capabilities.slice(0, 3).map((capability) => (
              <CapabilityBadge 
                key={capability} 
                capability={capability}
                type={capability.toLowerCase().includes('vision') ? 'primary' : 'secondary'}
              />
            ))}
            {model.capabilities.length > 3 && (
              <Badge color="gray" variant="outline" className="text-xs">
                +{model.capabilities.length - 3}
              </Badge>
            )}
          </div>
        </div>
      )}

      {/* Use cases */}
      {model.useCase.length > 0 && (
        <div className="mb-3">
          <div className="flex flex-wrap gap-1">
            {model.useCase.slice(0, 2).map((useCase) => (
              <Badge key={useCase} color="gray" variant="outline" className="text-xs">
                {useCase}
              </Badge>
            ))}
            {model.useCase.length > 2 && (
              <span className="text-xs text-gray-500">+{model.useCase.length - 2}</span>
            )}
          </div>
        </div>
      )}

      {/* Action button */}
      {getActionButton()}

      {/* Selection indicator */}
      {isSelected && (
        <div className={`absolute bottom-0 left-0 right-0 h-1 rounded-b-lg bg-${providerColor}-500/50 border-t border-${providerColor}-500/40`} />
      )}
    </motion.div>
  );
};