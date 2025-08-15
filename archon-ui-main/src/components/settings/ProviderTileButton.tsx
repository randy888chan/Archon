import React from 'react';
import { Check, Diamond, Zap } from 'lucide-react';

// Provider icon components
const OpenAIIcon = () => (
  <svg viewBox="0 0 24 24" className="w-6 h-6" fill="currentColor">
    <path d="M22.2819 9.8211a5.9847 5.9847 0 0 0-.5157-4.9108 6.0462 6.0462 0 0 0-6.5098-2.9A6.0651 6.0651 0 0 0 4.9807 4.1818a5.9847 5.9847 0 0 0-3.9977 2.9 6.0462 6.0462 0 0 0 .7427 7.0966 5.98 5.98 0 0 0 .511 4.9107 6.051 6.051 0 0 0 6.5146 2.9001A5.9847 5.9847 0 0 0 13.2599 24a6.0557 6.0557 0 0 0 5.7718-4.2058 5.9894 5.9894 0 0 0 3.9977-2.9001 6.0557 6.0557 0 0 0-.7475-7.0729zm-9.022 12.6081a4.4755 4.4755 0 0 1-2.8764-1.0408l.1419-.0804 4.7783-2.7582a.7948.7948 0 0 0 .3927-.6813v-6.7369l2.02 1.1686a.071.071 0 0 1 .038.052v5.5826a4.504 4.504 0 0 1-4.4945 4.4944zm-9.6607-4.1254a4.4708 4.4708 0 0 1-.5346-3.0137l.142.0852 4.783 2.7582a.7712.7712 0 0 0 .7806 0l5.8428-3.3685v2.3324a.0804.0804 0 0 1-.0332.0615L9.74 19.9502a4.4992 4.4992 0 0 1-6.1408-1.6464zM2.3408 7.8956a4.485 4.485 0 0 1 2.3655-1.9728V11.6a.7664.7664 0 0 0 .3879.6765l5.8144 3.3543-2.0201 1.1685a.0757.0757 0 0 1-.071 0l-4.8303-2.7865A4.504 4.504 0 0 1 2.3408 7.872zm16.5963 3.8558L13.1038 8.364 15.1192 7.2a.0757.0757 0 0 1 .071 0l4.8303 2.7913a4.4944 4.4944 0 0 1-.6765 8.1042v-5.6772a.79.79 0 0 0-.407-.667zm2.0107-3.0231l-.142-.0852-4.7735-2.7818a.7759.7759 0 0 0-.7854 0L9.409 9.2297V6.8974a.0662.0662 0 0 1 .0284-.0615l4.8303-2.7866a4.4992 4.4992 0 0 1 6.6802 4.66zM8.3065 12.863l-2.02-1.1638a.0804.0804 0 0 1-.038-.0567V6.0742a4.4992 4.4992 0 0 1 7.3757-3.4537l-.142.0805L8.704 5.459a.7948.7948 0 0 0-.3927.6813zm1.0976-2.3654l2.602-1.4998 2.6069 1.4998v2.9994l-2.5974 1.4997-2.6067-1.4997Z"/>
  </svg>
);

const GoogleGeminiIcon = () => (
  <Diamond className="w-6 h-6" />
);

const OllamaIcon = () => (
  <svg viewBox="0 0 24 24" className="w-6 h-6" fill="currentColor">
    <path d="M12 2L2 7L12 12L22 7L12 2Z" />
    <path d="M2 17L12 22L22 17" />
    <path d="M2 12L12 17L22 12" />
  </svg>
);

const AnthropicIcon = () => (
  <svg viewBox="0 0 24 24" className="w-6 h-6" fill="currentColor">
    <path d="M8 3L4 21H6.5L7.5 18H12.5L13.5 21H16L12 3H8ZM8.5 16L10 11L11.5 16H8.5Z" />
    <path d="M14.5 3V21H17V14H20V12H17V5H20V3H14.5Z" />
  </svg>
);

export type Provider = 'openai' | 'google' | 'ollama' | 'anthropic';

interface ProviderTileButtonProps {
  provider: Provider;
  title: string;
  description: string;
  isSelected: boolean;
  onClick: () => void;
  disabled?: boolean;
  badge?: string;
  className?: string;
}

const providerConfig = {
  openai: {
    icon: OpenAIIcon,
    accentColor: 'green',
    colors: {
      border: 'border-green-500',
      background: 'bg-green-500/10',
      glow: 'shadow-[0_0_20px_rgba(34,197,94,0.3)]',
      text: 'text-green-400'
    }
  },
  google: {
    icon: GoogleGeminiIcon,
    accentColor: 'blue',
    colors: {
      border: 'border-blue-500',
      background: 'bg-blue-500/10',
      glow: 'shadow-[0_0_20px_rgba(59,130,246,0.3)]',
      text: 'text-blue-400'
    }
  },
  ollama: {
    icon: OllamaIcon,
    accentColor: 'orange',
    colors: {
      border: 'border-orange-500',
      background: 'bg-orange-500/10',
      glow: 'shadow-[0_0_20px_rgba(249,115,22,0.3)]',
      text: 'text-orange-400'
    }
  },
  anthropic: {
    icon: AnthropicIcon,
    accentColor: 'purple',
    colors: {
      border: 'border-purple-500',
      background: 'bg-purple-500/10',
      glow: 'shadow-[0_0_20px_rgba(168,85,247,0.3)]',
      text: 'text-purple-400'
    }
  }
} as const;

export const ProviderTileButton: React.FC<ProviderTileButtonProps> = ({
  provider,
  title,
  description,
  isSelected,
  onClick,
  disabled = false,
  badge,
  className = ''
}) => {
  const config = providerConfig[provider];
  const IconComponent = config.icon;

  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={`
        relative p-4 rounded-lg border-2 transition-all duration-300
        bg-gradient-to-b from-white/80 to-white/60 dark:from-white/5 dark:to-black/20
        hover:shadow-lg hover:scale-[1.02]
        focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-offset-transparent
        disabled:opacity-50 disabled:cursor-not-allowed
        ${isSelected 
          ? `${config.colors.border} ${config.colors.background} ${config.colors.glow}` 
          : 'border-gray-300 dark:border-gray-600 hover:border-gray-400 dark:hover:border-gray-500'
        }
        ${disabled ? 'grayscale' : ''}
        ${className}
      `}
    >
      {/* Selection indicator */}
      {isSelected && (
        <div className="absolute top-2 right-2">
          <div className={`
            w-6 h-6 rounded-full flex items-center justify-center
            ${config.colors.background} ${config.colors.border} border
          `}>
            <Check className={`w-4 h-4 ${config.colors.text}`} />
          </div>
        </div>
      )}

      {/* Badge (e.g., "Soon") */}
      {badge && (
        <div className="absolute top-2 right-2">
          <span className="px-2 py-1 text-xs font-medium rounded-full bg-gray-500/20 text-gray-600 dark:text-gray-400 border border-gray-500/30">
            {badge}
          </span>
        </div>
      )}

      {/* Provider icon */}
      <div className={`
        mb-3 flex items-center justify-center
        ${isSelected ? config.colors.text : 'text-gray-600 dark:text-gray-400'}
      `}>
        <IconComponent />
      </div>

      {/* Provider title */}
      <h3 className={`
        font-semibold text-sm mb-2
        ${isSelected ? config.colors.text : 'text-gray-900 dark:text-white'}
      `}>
        {title}
      </h3>

      {/* Provider description */}
      <p className="text-xs text-gray-600 dark:text-gray-400 leading-tight">
        {description}
      </p>

      {/* Selected state indicator line */}
      {isSelected && (
        <div className={`
          absolute bottom-0 left-0 right-0 h-1 rounded-b-lg
          ${config.colors.background} ${config.colors.border} border-t
        `} />
      )}
    </button>
  );
};

// Provider data for easy consumption
export const PROVIDERS = {
  openai: {
    id: 'openai' as Provider,
    title: 'OpenAI',
    description: 'Industry-leading AI models with GPT-4 and excellent tool support'
  },
  google: {
    id: 'google' as Provider,
    title: 'Google Gemini',
    description: "Google's powerful multimodal AI with fast inference"
  },
  ollama: {
    id: 'ollama' as Provider,
    title: 'Ollama',
    description: 'Run open-source models locally with privacy and control'
  },
  anthropic: {
    id: 'anthropic' as Provider,
    title: 'Anthropic',
    description: 'Claude models with excellent reasoning and safety'
  }
} as const;