# ProviderTileButton Component

A reusable React component for displaying AI provider selection tiles with modern UI design, matching the Archon V2 Alpha interface specifications.

## Features

- **Provider Support**: OpenAI, Google Gemini, Ollama, Anthropic
- **Visual States**: Selected/unselected with color-coded borders and backgrounds
- **Accessibility**: Full keyboard navigation and ARIA support
- **Dark Mode**: Complete dark mode compatibility
- **Responsive**: Works across all screen sizes
- **Customizable**: Badge support, disabled states, custom styling

## Usage

### Basic Usage

```tsx
import { ProviderTileButton, Provider } from './ProviderTileButton';

const MyComponent = () => {
  const [selectedProvider, setSelectedProvider] = useState<Provider>('openai');

  return (
    <ProviderTileButton
      provider="openai"
      title="OpenAI"
      description="Industry-leading AI models with GPT-4 and excellent tool support"
      isSelected={selectedProvider === 'openai'}
      onClick={() => setSelectedProvider('openai')}
    />
  );
};
```

### Grid Layout with ProviderSelectionGrid

```tsx
import { ProviderSelectionGrid } from './ProviderSelectionGrid';

const ProviderSettings = () => {
  const [llmProvider, setLlmProvider] = useState<Provider>('openai');
  const [embeddingProvider, setEmbeddingProvider] = useState<Provider>('openai');

  return (
    <>
      <ProviderSelectionGrid
        selectedProvider={llmProvider}
        onProviderSelect={setLlmProvider}
        title="LLM Provider Selection"
        subtitle="Choose your language model provider"
      />
      
      <ProviderSelectionGrid
        selectedProvider={embeddingProvider}
        onProviderSelect={setEmbeddingProvider}
        title="Embedding Provider Selection"
        subtitle="Choose your embedding provider"
        disabledProviders={['anthropic']} // Anthropic has no embedding models
      />
    </>
  );
};
```

## Props

### ProviderTileButton Props

| Prop | Type | Default | Description |
|------|------|---------|-------------|
| `provider` | `Provider` | - | Provider identifier ('openai', 'google', 'ollama', 'anthropic') |
| `title` | `string` | - | Display title for the provider |
| `description` | `string` | - | Description text shown below the title |
| `isSelected` | `boolean` | - | Whether this provider is currently selected |
| `onClick` | `() => void` | - | Handler called when tile is clicked |
| `disabled` | `boolean` | `false` | Whether the tile is disabled |
| `badge` | `string` | - | Optional badge text (e.g., "Soon", "Beta") |
| `className` | `string` | `''` | Additional CSS classes |

### ProviderSelectionGrid Props

| Prop | Type | Default | Description |
|------|------|---------|-------------|
| `selectedProvider` | `Provider` | - | Currently selected provider |
| `onProviderSelect` | `(provider: Provider) => void` | - | Handler for provider selection |
| `title` | `string` | - | Grid section title |
| `subtitle` | `string` | - | Optional subtitle |
| `className` | `string` | `''` | Additional CSS classes |
| `disabledProviders` | `Provider[]` | `[]` | Array of providers to disable |

## Provider Types

```typescript
type Provider = 'openai' | 'google' | 'ollama' | 'anthropic';
```

## Styling

The component uses Tailwind CSS classes and follows the Archon design system:

- **Selection States**: Green accent for selected providers
- **Provider Colors**: Each provider has its own accent color (green, blue, orange, purple)
- **Dark Mode**: Automatic dark mode support through dark: variants
- **Hover Effects**: Scale and shadow transitions on hover
- **Accessibility**: Focus rings and proper contrast ratios

## Provider Configuration

Each provider has predefined colors and icons:

- **OpenAI**: Green accent with OpenAI logo
- **Google Gemini**: Blue accent with diamond icon
- **Ollama**: Orange accent with layers icon (representing local models)
- **Anthropic**: Purple accent with stylized "A" logo

## Integration with RAGSettings

To integrate into existing settings components:

1. Import the components
2. Replace dropdown selectors with `ProviderSelectionGrid`
3. Update state management to use `Provider` type
4. Add provider-specific configuration sections

See `RAGSettingsWithTiles.tsx` for a complete integration example.

## Accessibility

- Full keyboard navigation support
- Proper ARIA labels and roles
- High contrast ratios in both light and dark modes
- Screen reader friendly descriptions
- Focus indicators for keyboard users

## Browser Support

- Modern browsers with CSS Grid support
- React 18+
- TypeScript support included