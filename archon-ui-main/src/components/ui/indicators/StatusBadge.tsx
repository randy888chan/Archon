import React from 'react';
import { CheckCircle2, AlertCircle, Loader2, RefreshCw, Download } from 'lucide-react';
import { Badge } from '../Badge';
import { Button } from '../Button';

interface StatusBadgeProps {
  status: 'available' | 'installing' | 'error' | 'unavailable';
  className?: string;
  showRetry?: boolean;
  onRetry?: () => void;
  size?: 'sm' | 'md' | 'lg';
  variant?: 'badge' | 'indicator';
}

export const StatusBadge: React.FC<StatusBadgeProps> = ({
  status,
  className = '',
  showRetry = false,
  onRetry,
  size = 'md',
  variant = 'badge'
}) => {
  const statusConfig = {
    available: {
      icon: <CheckCircle2 className={`${size === 'sm' ? 'w-3 h-3' : size === 'lg' ? 'w-5 h-5' : 'w-4 h-4'} text-green-500`} />,
      label: 'Available',
      color: 'green' as const,
      bgColor: 'bg-green-500/10',
      description: 'Model is ready for use'
    },
    installing: {
      icon: <Loader2 className={`${size === 'sm' ? 'w-3 h-3' : size === 'lg' ? 'w-5 h-5' : 'w-4 h-4'} text-blue-500 animate-spin`} />,
      label: 'Installing',
      color: 'blue' as const,
      bgColor: 'bg-blue-500/10',
      description: 'Model is currently being downloaded/installed'
    },
    error: {
      icon: <AlertCircle className={`${size === 'sm' ? 'w-3 h-3' : size === 'lg' ? 'w-5 h-5' : 'w-4 h-4'} text-red-500`} />,
      label: 'Error',
      color: 'pink' as const,
      bgColor: 'bg-red-500/10',
      description: 'Error occurred during installation or connection'
    },
    unavailable: {
      icon: <Download className={`${size === 'sm' ? 'w-3 h-3' : size === 'lg' ? 'w-5 h-5' : 'w-4 h-4'} text-gray-400`} />,
      label: 'Not Installed',
      color: 'gray' as const,
      bgColor: 'bg-gray-500/10',
      description: 'Model needs to be downloaded/installed'
    }
  };

  const config = statusConfig[status];
  
  const sizeMap = {
    sm: 'text-xs px-1.5 py-0.5',
    md: 'text-sm px-2 py-1',
    lg: 'text-base px-3 py-1.5'
  };

  if (variant === 'indicator') {
    return (
      <div 
        className={`flex items-center gap-2 ${className}`}
        title={config.description}
      >
        <div className={`p-1 rounded-full ${config.bgColor}`}>
          {config.icon}
        </div>
        <span className={`font-medium text-${config.color === 'pink' ? 'red' : config.color}-500`}>
          {config.label}
        </span>
        {showRetry && status === 'error' && onRetry && (
          <Button
            size="sm"
            variant="ghost"
            onClick={onRetry}
            className="h-6 px-2 ml-1"
          >
            <RefreshCw className="w-3 h-3" />
          </Button>
        )}
      </div>
    );
  }

  return (
    <div className={`flex items-center gap-2 ${className}`}>
      <Badge 
        color={config.color} 
        variant="outline" 
        className={`${sizeMap[size]} flex items-center gap-1.5`}
        title={config.description}
      >
        {config.icon}
        <span>{config.label}</span>
      </Badge>
      {showRetry && status === 'error' && onRetry && (
        <Button
          size="sm"
          variant="ghost"
          onClick={onRetry}
          className="h-6 px-2"
        >
          <RefreshCw className="w-3 h-3" />
        </Button>
      )}
    </div>
  );
};