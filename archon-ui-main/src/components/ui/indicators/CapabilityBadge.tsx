import React from 'react';
import {
  Eye,
  Zap,
  Layers,
  Database,
  Cpu,
  Users,
  MessageSquare,
  Image,
  Code,
  FileText,
  Brain,
  Sparkles
} from 'lucide-react';
import { Badge } from '../Badge';

interface CapabilityBadgeProps {
  capability: string;
  className?: string;
  type?: 'primary' | 'secondary' | 'success' | 'warning';
  size?: 'sm' | 'md' | 'lg';
  showIcon?: boolean;
  description?: string;
}

export const CapabilityBadge: React.FC<CapabilityBadgeProps> = ({
  capability,
  className = '',
  type = 'secondary',
  size = 'md',
  showIcon = true,
  description
}) => {
  const getIconForCapability = (cap: string) => {
    const lowerCap = cap.toLowerCase();
    
    // Vision and image capabilities
    if (lowerCap.includes('vision') || lowerCap.includes('image') || lowerCap.includes('visual')) {
      return <Eye className="w-3 h-3" />;
    }
    
    // Function and tool capabilities
    if (lowerCap.includes('function') || lowerCap.includes('tool') || lowerCap.includes('api')) {
      return <Zap className="w-3 h-3" />;
    }
    
    // Code capabilities
    if (lowerCap.includes('code') || lowerCap.includes('programming') || lowerCap.includes('development')) {
      return <Code className="w-3 h-3" />;
    }
    
    // Text and language capabilities
    if (lowerCap.includes('text') || lowerCap.includes('language') || lowerCap.includes('nlp')) {
      return <FileText className="w-3 h-3" />;
    }
    
    // Embedding capabilities
    if (lowerCap.includes('embedding') || lowerCap.includes('vector') || lowerCap.includes('semantic')) {
      return <Layers className="w-3 h-3" />;
    }
    
    // Chat and conversation capabilities
    if (lowerCap.includes('chat') || lowerCap.includes('conversation') || lowerCap.includes('dialogue')) {
      return <MessageSquare className="w-3 h-3" />;
    }
    
    // Multimodal capabilities
    if (lowerCap.includes('multimodal') || lowerCap.includes('multi-modal')) {
      return <Sparkles className="w-3 h-3" />;
    }
    
    // Reasoning and intelligence capabilities
    if (lowerCap.includes('reasoning') || lowerCap.includes('intelligence') || lowerCap.includes('analysis')) {
      return <Brain className="w-3 h-3" />;
    }
    
    // Data and database capabilities
    if (lowerCap.includes('data') || lowerCap.includes('database') || lowerCap.includes('query')) {
      return <Database className="w-3 h-3" />;
    }
    
    // Processing capabilities
    if (lowerCap.includes('processing') || lowerCap.includes('compute') || lowerCap.includes('performance')) {
      return <Cpu className="w-3 h-3" />;
    }
    
    // Collaboration and multi-user capabilities
    if (lowerCap.includes('collaboration') || lowerCap.includes('team') || lowerCap.includes('multi-user')) {
      return <Users className="w-3 h-3" />;
    }

    // Default icon for unknown capabilities
    return <Sparkles className="w-3 h-3" />;
  };

  const colorMap = {
    primary: 'blue',
    secondary: 'gray', 
    success: 'green',
    warning: 'orange'
  } as const;

  const sizeMap = {
    sm: 'text-xs px-1.5 py-0.5',
    md: 'text-sm px-2 py-1',
    lg: 'text-base px-3 py-1.5'
  };

  const icon = showIcon ? getIconForCapability(capability) : null;
  const title = description || `${capability} capability`;

  return (
    <div className={className} title={title}>
      <Badge 
        color={colorMap[type]} 
        variant="outline" 
        className={`${sizeMap[size]} flex items-center gap-1.5`}
      >
        {icon}
        <span className="font-medium">{capability}</span>
      </Badge>
    </div>
  );
};