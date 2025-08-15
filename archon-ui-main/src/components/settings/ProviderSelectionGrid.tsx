import React from 'react';
import { ProviderTileButton, Provider, PROVIDERS } from './ProviderTileButton';

interface ProviderSelectionGridProps {
  selectedProvider: Provider;
  onProviderSelect: (provider: Provider) => void;
  title: string;
  subtitle?: string;
  className?: string;
  disabledProviders?: Provider[];
}

export const ProviderSelectionGrid: React.FC<ProviderSelectionGridProps> = ({
  selectedProvider,
  onProviderSelect,
  title,
  subtitle,
  className = '',
  disabledProviders = []
}) => {
  return (
    <div className={className}>
      <div className="flex items-center mb-4">
        <div className="w-6 h-6 rounded-full bg-green-500/20 border border-green-500/30 flex items-center justify-center mr-3">
          <div className="w-2 h-2 rounded-full bg-green-500" />
        </div>
        <div>
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
            {title}
          </h3>
          {subtitle && (
            <p className="text-sm text-gray-600 dark:text-gray-400">
              {subtitle}
            </p>
          )}
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {Object.values(PROVIDERS).map((provider) => (
          <ProviderTileButton
            key={provider.id}
            provider={provider.id}
            title={provider.title}
            description={provider.description}
            isSelected={selectedProvider === provider.id}
            onClick={() => onProviderSelect(provider.id)}
            disabled={disabledProviders.includes(provider.id)}
            badge={provider.id === 'anthropic' ? 'Soon' : undefined}
          />
        ))}
      </div>
    </div>
  );
};