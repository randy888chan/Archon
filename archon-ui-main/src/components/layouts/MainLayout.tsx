import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { SideNavigation } from './SideNavigation';
import { ArchonChatPanel } from './ArchonChatPanel';
import { X } from 'lucide-react';
import { useToast } from '../../contexts/ToastContext';
import { credentialsService } from '../../services/credentialsService';
/**
 * Props for the MainLayout component
 */
interface MainLayoutProps {
  children: React.ReactNode;
}
/**
 * MainLayout - The main layout component for the application
 *
 * This component provides the overall layout structure including:
 * - Side navigation
 * - Main content area
 * - Knowledge chat panel (slidable)
 */
export const MainLayout: React.FC<MainLayoutProps> = ({
  children
}) => {
  // State to track if chat panel is open
  const [isChatOpen, setIsChatOpen] = useState(false);
  const { showToast } = useToast();
  const navigate = useNavigate();
  const [backendReady, setBackendReady] = useState(false);

  // Check backend readiness
  useEffect(() => {
    
    const checkBackendHealth = async (retryCount = 0) => {
      const maxRetries = 10; // Increased retries for initialization
      const retryDelay = 1000;
      
      try {
        // Check if backend is responding with a simple health check
        const response = await fetch(`${credentialsService['baseUrl']}/health`, {
          method: 'GET',
          timeout: 5000
        } as any);
        
        if (response.ok) {
          const healthData = await response.json();
          console.log('📋 Backend health check:', healthData);
          
          // Check if backend is truly ready (not just started)
          if (healthData.ready === true) {
            console.log('✅ Backend is fully initialized');
            setBackendReady(true);
          } else {
            // Backend is starting up but not ready yet
            console.log(`🔄 Backend initializing... (attempt ${retryCount + 1}/${maxRetries}):`, healthData.message || 'Loading credentials...');
            
            // Retry with shorter interval during initialization
            if (retryCount < maxRetries) {
              setTimeout(() => {
                checkBackendHealth(retryCount + 1);
              }, retryDelay); // Constant 1s retry during initialization
            } else {
              console.warn('Backend initialization taking too long - skipping credential check');
              setBackendReady(false);
            }
          }
        } else {
          throw new Error(`Backend health check failed: ${response.status}`);
        }
      } catch (error) {
        const errorMessage = error instanceof Error ? error.message : 'Unknown error';
        console.log(`Backend not ready yet (attempt ${retryCount + 1}/${maxRetries}):`, errorMessage);
        
        // Retry if we haven't exceeded max retries
        if (retryCount < maxRetries) {
          setTimeout(() => {
            checkBackendHealth(retryCount + 1);
          }, retryDelay * Math.pow(1.5, retryCount)); // Exponential backoff for connection errors
        } else {
          console.warn('Backend not ready after maximum retries - skipping credential check');
          setBackendReady(false);
        }
      }
    };


    // Start the health check process
    setTimeout(() => {
      checkBackendHealth();
    }, 1000); // Wait 1 second for initial app startup
  }, [showToast, navigate]); // Removed backendReady from dependencies to prevent double execution

  return <div className="relative min-h-screen bg-white dark:bg-black overflow-hidden">
      {/* Fixed full-page background grid that doesn't scroll */}
      <div className="fixed inset-0 neon-grid pointer-events-none z-0"></div>
      {/* Floating Navigation */}
      <div className="fixed left-6 top-1/2 -translate-y-1/2 z-50">
        <SideNavigation />
      </div>
      {/* Main Content Area - no left margin to allow grid to extend full width */}
      <div className="relative flex-1 pl-[100px] z-10">
        <div className="container mx-auto px-8 relative">
          <div className="min-h-screen pt-8 pb-16">{children}</div>
        </div>
      </div>
      {/* Floating Chat Button - Only visible when chat is closed */}
      {!isChatOpen && <button onClick={() => setIsChatOpen(true)} className="fixed bottom-6 right-6 z-50 w-14 h-14 rounded-full flex items-center justify-center backdrop-blur-md bg-gradient-to-b from-white/10 to-black/30 dark:from-white/10 dark:to-black/30 from-blue-100/80 to-blue-50/60 shadow-[0_0_20px_rgba(59,130,246,0.3)] dark:shadow-[0_0_20px_rgba(59,130,246,0.7)] hover:shadow-[0_0_25px_rgba(59,130,246,0.5)] dark:hover:shadow-[0_0_25px_rgba(59,130,246,0.9)] transition-all duration-300 overflow-hidden border border-blue-200 dark:border-transparent" aria-label="Open Knowledge Assistant">
          <img src="/logo-neon.svg" alt="Archon" className="w-7 h-7" />
        </button>}
      {/* Chat Sidebar - Slides in/out from right */}
      <div className="fixed top-0 right-0 h-full z-40 transition-transform duration-300 ease-in-out transform" style={{
      transform: isChatOpen ? 'translateX(0)' : 'translateX(100%)'
    }}>
        {/* Close button - Only visible when chat is open */}
        {isChatOpen && <button onClick={() => setIsChatOpen(false)} className="absolute -left-14 bottom-6 z-50 w-12 h-12 rounded-full flex items-center justify-center backdrop-blur-md bg-gradient-to-b from-white/10 to-black/30 dark:from-white/10 dark:to-black/30 from-pink-100/80 to-pink-50/60 border border-pink-200 dark:border-pink-500/30 shadow-[0_0_15px_rgba(236,72,153,0.2)] dark:shadow-[0_0_15px_rgba(236,72,153,0.5)] hover:shadow-[0_0_20px_rgba(236,72,153,0.4)] dark:hover:shadow-[0_0_20px_rgba(236,72,153,0.7)] transition-all duration-300" aria-label="Close Knowledge Assistant">
            <X className="w-5 h-5 text-pink-500" />
          </button>}
        {/* Knowledge Chat Panel */}
        <ArchonChatPanel data-id="archon-chat" />
      </div>
    </div>;
};