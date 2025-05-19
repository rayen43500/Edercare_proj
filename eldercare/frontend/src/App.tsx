import React, { useState, useEffect, useRef } from 'react';
import './App.css';
import elderCareApi from './services/eldercare-api';

function App() {
  const [message, setMessage] = useState('');
  const [conversation, setConversation] = useState<{text: string, sender: string, speaking?: boolean}[]>([
    {text: "Bonjour, je suis votre assistant ElderCare. Comment puis-je vous aider aujourd'hui?", sender: 'ai', speaking: false}
  ]);
  const [isRecording, setIsRecording] = useState(false);
  const conversationEndRef = useRef<HTMLDivElement>(null);
  const [isTyping, setIsTyping] = useState(false);
  const [showFloatingMic, setShowFloatingMic] = useState(false);
  const [apiInitialized, setApiInitialized] = useState(false);
  const [initializationError, setInitializationError] = useState<string | null>(null);
  const [retryCount, setRetryCount] = useState(0);
  const maxRetries = 3;

  // Initialiser l'API au chargement
  useEffect(() => {
    const initializeApi = async () => {
      try {
        setIsTyping(true);
        
        // Vérifier si l'API est déjà initialisée
        const healthCheck = await elderCareApi.checkHealth();
        
        if (healthCheck.status === "error") {
          console.error("Échec de la vérification de l'état de l'API");
          setInitializationError("Le serveur ne répond pas. Veuillez réessayer plus tard.");
          setApiInitialized(false);
          return;
        }
        
        if (!healthCheck.initialized) {
          const initResult = await elderCareApi.initializeAssistant();
          setApiInitialized(initResult.status === 'success');
          
          if (initResult.status !== 'success') {
            setInitializationError(initResult.message || 'Erreur d\'initialisation');
          }
        } else {
          setApiInitialized(true);
          console.log("API déjà initialisée");
        }
      } catch (error) {
        console.error('Erreur lors de la connexion à l\'API:', error);
        setInitializationError('Impossible de se connecter à l\'API ElderCare');
      } finally {
        setIsTyping(false);
      }
    };

    initializeApi();
    
    // Retry logic
    const retryInterval = setInterval(() => {
      if (!apiInitialized && retryCount < maxRetries) {
        setRetryCount(prev => prev + 1);
        console.log(`Tentative de reconnexion ${retryCount + 1}/${maxRetries}...`);
        initializeApi();
      } else {
        clearInterval(retryInterval);
      }
    }, 5000);
    
    return () => clearInterval(retryInterval);
  }, [apiInitialized, retryCount]);

  // Scroll to bottom of conversation when messages are added
  useEffect(() => {
    if (conversationEndRef.current) {
      conversationEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [conversation]);

  // Show floating mic when scrolling down
  useEffect(() => {
    const handleScroll = () => {
      if (document.querySelector('.conversation')) {
        const conversationEl = document.querySelector('.conversation') as HTMLElement;
        if (conversationEl.scrollTop > 100) {
          setShowFloatingMic(true);
        } else {
          setShowFloatingMic(false);
        }
      }
    };

    const conversationEl = document.querySelector('.conversation');
    if (conversationEl) {
      conversationEl.addEventListener('scroll', handleScroll);
    }

    return () => {
      if (conversationEl) {
        conversationEl.removeEventListener('scroll', handleScroll);
      }
    };
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (message.trim() === '') return;
    
    // Add user message to conversation
    const userMessage = message;
    setConversation(prev => [...prev, {text: userMessage, sender: 'user'}]);
    
    // Clear input
    setMessage('');
    
    // Show AI is typing indicator
    setIsTyping(true);
    
    try {
      // Send message to API
      const response = await elderCareApi.sendMessage(userMessage);
      
      if (response.status === 'success') {
        // Add AI response to conversation
        const aiMessage = {text: response.response, sender: 'ai', speaking: true};
        setConversation(prev => [...prev, aiMessage]);
        
        // Stop speaking animation after 3 seconds
        setTimeout(() => {
          setConversation(prev => 
            prev.map(msg => 
              msg === aiMessage ? {...msg, speaking: false} : msg
            )
          );
        }, 3000);
        
        // If API was not initialized before, it is now
        if (!apiInitialized) {
          setApiInitialized(true);
          setInitializationError(null);
        }
      } else {
        // Handle error in response
        throw new Error(response.message || "Erreur lors du traitement du message");
      }
    } catch (error) {
      console.error('Erreur lors de l\'envoi du message:', error);
      
      // Add error message
      let errorMessage = "Désolé, je n'ai pas pu traiter votre message. Veuillez réessayer plus tard.";
      
      if (error instanceof Error) {
        errorMessage = error.message;
      }
      
      setConversation(prev => [...prev, {
        text: errorMessage,
        sender: 'ai'
      }]);
      
      // If we get an error, check API health
      try {
        const healthCheck = await elderCareApi.checkHealth();
        setApiInitialized(healthCheck.initialized);
      } catch (e) {
        setApiInitialized(false);
      }
    } finally {
      setIsTyping(false);
    }
  };

  const toggleRecording = () => {
    setIsRecording(!isRecording);
    
    if (!isRecording) {
      // Simulate recording ending after 3 seconds
      setTimeout(() => {
        setIsRecording(false);
        
        // Add simulated voice message from user
        const voiceMessage = "Voici mon message vocal pour l'assistant...";
        setConversation(prev => [...prev, {text: voiceMessage, sender: 'user'}]);
        
        // Process message as if it was typed
        handleVoiceMessage(voiceMessage);
      }, 3000);
    }
  };
  
  const handleVoiceMessage = async (voiceText: string) => {
    // Show AI is typing indicator
    setIsTyping(true);
    
    try {
      // Send message to API
      const response = await elderCareApi.sendMessage(voiceText);
      
      if (response.status === 'success') {
        // Add AI response to conversation
        const aiMessage = {text: response.response, sender: 'ai', speaking: true};
        setConversation(prev => [...prev, aiMessage]);
        
        // Stop speaking animation after 3 seconds
        setTimeout(() => {
          setConversation(prev => 
            prev.map(msg => 
              msg === aiMessage ? {...msg, speaking: false} : msg
            )
          );
        }, 3000);
      } else {
        throw new Error(response.message || "Erreur lors du traitement du message vocal");
      }
    } catch (error) {
      console.error('Erreur lors du traitement du message vocal:', error);
      
      // Add error message
      let errorMessage = "Désolé, je n'ai pas pu traiter votre message vocal. Veuillez réessayer plus tard.";
      
      if (error instanceof Error) {
        errorMessage = error.message;
      }
      
      setConversation(prev => [...prev, {
        text: errorMessage,
        sender: 'ai'
      }]);
    } finally {
      setIsTyping(false);
    }
  };

  const retryInitialization = async () => {
    setInitializationError(null);
    setIsTyping(true);
    
    try {
      const initResult = await elderCareApi.initializeAssistant();
      setApiInitialized(initResult.status === 'success');
      
      if (initResult.status !== 'success') {
        setInitializationError(initResult.message || 'Erreur d\'initialisation');
      } else {
        // Add success message
        setConversation(prev => [...prev, {
          text: "Je suis maintenant connecté et prêt à vous aider !",
          sender: 'ai'
        }]);
      }
    } catch (error) {
      console.error('Erreur lors de la reconnexion à l\'API:', error);
      setInitializationError('Impossible de se connecter à l\'API ElderCare');
    } finally {
      setIsTyping(false);
    }
  };

  return (
    <div className={`app-container ${showFloatingMic ? 'show-floating-mic' : ''}`}>
      {/* Background bubbles */}
      <div className="background-bubbles">
        <div className="bubble"></div>
        <div className="bubble"></div>
        <div className="bubble"></div>
        <div className="bubble"></div>
        <div className="bubble"></div>
      </div>
      
      <div className="app-header">
        <div className="tech-particles">
          <div className="tech-circle"></div>
          <div className="tech-circle"></div>
          <div className="tech-circle"></div>
          <div className="tech-line"></div>
          <div className="tech-line"></div>
        </div>
        <h1>ElderCare</h1>
        <p className="app-subtitle">Assistant IA pour le soin des personnes âgées</p>
        {initializationError && (
          <div className="error-banner">
            {initializationError}
            <button onClick={retryInitialization} className="retry-button">
              Réessayer
            </button>
          </div>
        )}
      </div>
      
      <div className="chat-container">
        <div className="conversation">
          {conversation.map((entry, index) => (
            <div key={index} className={`message ${entry.sender} ${entry.speaking ? 'speaking' : ''}`}>
              <div className="message-bubble">
                {entry.text}
                {entry.speaking && (
                  <div className="voice-animation">
                    <div className="voice-bar"></div>
                    <div className="voice-bar"></div>
                    <div className="voice-bar"></div>
                    <div className="voice-bar"></div>
                    <div className="voice-bar"></div>
                  </div>
                )}
              </div>
            </div>
          ))}
          {isTyping && (
            <div className="message ai">
              <div className="message-bubble typing-indicator">
                <span></span>
                <span></span>
                <span></span>
              </div>
            </div>
          )}
          <div ref={conversationEndRef} />
        </div>
        
        {/* Floating Google Assistant style microphone */}
        <div className="voice-fab-wrapper">
          <button 
            className="floating-voice-button" 
            onClick={toggleRecording}
            aria-label="Activer l'assistant vocal"
            disabled={isTyping || !apiInitialized}
          >
            <div className="indicator-dot"></div>
            <div className="floating-voice-ripple"></div>
            <div className="floating-voice-ripple"></div>
            <div className="floating-voice-ripple"></div>
            <div className="floating-voice-ripple"></div>
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M12 16C14.2091 16 16 14.2091 16 12V6C16 3.79086 14.2091 2 12 2C9.79086 2 8 3.79086 8 6V12C8 14.2091 9.79086 16 12 16Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              <path d="M19 10V12C19 15.866 15.866 19 12 19C8.13401 19 5 15.866 5 12V10" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </button>
        </div>
        
        <form onSubmit={handleSubmit} className="message-form">
          <input
            type="text"
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            placeholder="Écrivez votre message ici..."
            className="message-input"
            disabled={isTyping || !apiInitialized}
          />
          <button 
            type="button" 
            className={`voice-button ${isRecording ? 'recording' : ''}`}
            onClick={toggleRecording}
            aria-label="Enregistrer un message vocal"
            disabled={isTyping || !apiInitialized}
          >
            <div className="voice-waves-container">
              <div className="voice-wave"></div>
              <div className="voice-wave"></div>
              <div className="voice-wave"></div>
              <div className="voice-wave"></div>
            </div>
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M12 16C14.2091 16 16 14.2091 16 12V6C16 3.79086 14.2091 2 12 2C9.79086 2 8 3.79086 8 6V12C8 14.2091 9.79086 16 12 16Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              <path d="M19 10V12C19 15.866 15.866 19 12 19C8.13401 19 5 15.866 5 12V10" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              <path d="M12 19V22" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              <path d="M8 22H16" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </button>
          <button type="submit" className="send-button" disabled={message.trim() === '' || isTyping || !apiInitialized}>
            <svg width="20" height="20" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M22 2L11 13" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              <path d="M22 2L15 22L11 13L2 9L22 2Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </button>
        </form>
      </div>
      
      <div className="features-section">
        <div className="feature-card">
          <div className="feature-icon">
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M22 12H18L15 21L9 3L6 12H2" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </div>
          <h3>Conseils Personnalisés</h3>
          <p>Recommandations adaptées aux besoins spécifiques</p>
        </div>
        <div className="feature-card">
          <div className="feature-icon">
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M22 12H18L15 21L9 3L6 12H2" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
              <path d="M12 7.5C13.3807 7.5 14.5 6.38071 14.5 5C14.5 3.61929 13.3807 2.5 12 2.5C10.6193 2.5 9.5 3.61929 9.5 5C9.5 6.38071 10.6193 7.5 12 7.5Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </div>
          <h3>Surveillance Santé</h3>
          <p>Suivi des signes vitaux et rappels de médicaments</p>
        </div>
        <div className="feature-card">
          <div className="feature-icon">
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
              <path d="M20.84 4.61C20.3293 4.09931 19.7228 3.69365 19.0554 3.41708C18.3879 3.14052 17.6725 2.99817 16.95 2.99817C16.2275 2.99817 15.5121 3.14052 14.8446 3.41708C14.1772 3.69365 13.5707 4.09931 13.06 4.61L12 5.67L10.94 4.61C9.9083 3.57831 8.50903 2.99715 7.05 2.99715C5.59096 2.99715 4.19169 3.57831 3.16 4.61C2.12831 5.64169 1.54716 7.04097 1.54716 8.5C1.54716 9.95903 2.12831 11.3583 3.16 12.39L4.22 13.45L12 21.23L19.78 13.45L20.84 12.39C21.3507 11.8793 21.7563 11.2728 22.0329 10.6054C22.3095 9.9379 22.4518 9.22249 22.4518 8.5C22.4518 7.77751 22.3095 7.0621 22.0329 6.3946C21.7563 5.72721 21.3507 5.12069 20.84 4.61V4.61Z" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
            </svg>
          </div>
          <h3>Soutien Émotionnel</h3>
          <p>Conversation et compagnie virtuelle</p>
        </div>
      </div>
    </div>
  );
}

export default App;
