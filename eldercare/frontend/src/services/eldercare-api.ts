/**
 * Service pour communiquer avec l'API ElderCare
 */

const API_BASE_URL = process.env.REACT_APP_API_BASE_URL || 'http://localhost:5000/api';

interface UserProfile {
  id: string;
  name: string;
  age: number;
  interests: string[];
  health_conditions: string[];
  tech_comfort: string;
}

interface MessageResponse {
  status: string;
  response: string;
  message?: string;
  conversation?: Array<{role: string; content: string}>;
}

interface ApiResponse<T> {
  status: string;
  message?: string;
  [key: string]: any;
}

// Configuration des requêtes fetch avec timeout
async function fetchWithTimeout(url: string, options: RequestInit = {}, timeout = 10000): Promise<Response> {
  const controller = new AbortController();
  const id = setTimeout(() => controller.abort(), timeout);
  
  const response = await fetch(url, {
    ...options,
    signal: controller.signal
  });
  
  clearTimeout(id);
  
  return response;
}

/**
 * Vérifie si l'API est accessible et initialisée
 */
export const checkHealth = async (): Promise<{status: string; initialized: boolean}> => {
  try {
    const response = await fetchWithTimeout(`${API_BASE_URL}/health`);
    return await response.json();
  } catch (error) {
    console.error("Erreur lors de la vérification de l'état de l'API:", error);
    return { status: "error", initialized: false };
  }
};

/**
 * Initialise l'assistant ElderCare
 */
export const initializeAssistant = async (): Promise<ApiResponse<any>> => {
  try {
    const response = await fetchWithTimeout(`${API_BASE_URL}/initialize`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      }
    }, 20000); // Plus long timeout pour l'initialisation
    
    return await response.json();
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") {
      return { status: "error", message: "L'initialisation a pris trop de temps. Veuillez réessayer." };
    }
    return { status: "error", message: "Impossible de connecter au serveur ElderCare" };
  }
};

/**
 * Envoie un message à l'assistant et récupère sa réponse
 */
export const sendMessage = async (message: string): Promise<MessageResponse> => {
  try {
    const response = await fetchWithTimeout(`${API_BASE_URL}/message`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ message })
    }, 15000);
    
    const data = await response.json();
    
    if (!response.ok) {
      throw new Error(data.message || "Erreur lors de l'envoi du message");
    }
    
    return data;
  } catch (error) {
    if (error instanceof DOMException && error.name === "AbortError") {
      throw new Error("La demande a pris trop de temps. Veuillez réessayer.");
    }
    if (error instanceof Error) {
      throw error;
    }
    throw new Error("Erreur inattendue lors de la communication avec le serveur");
  }
};

/**
 * Récupère le profil utilisateur
 */
export const getUserProfile = async (): Promise<ApiResponse<UserProfile>> => {
  try {
    const response = await fetchWithTimeout(`${API_BASE_URL}/user_profile`);
    const data = await response.json();
    
    if (!response.ok) {
      throw new Error(data.message || "Erreur lors de la récupération du profil utilisateur");
    }
    
    return data;
  } catch (error) {
    console.error("Erreur de profil utilisateur:", error);
    if (error instanceof Error) {
      throw error;
    }
    throw new Error("Erreur inattendue lors de la récupération du profil utilisateur");
  }
};

/**
 * Met à jour le profil utilisateur
 */
export const updateUserProfile = async (profile: Partial<UserProfile>): Promise<ApiResponse<UserProfile>> => {
  try {
    const response = await fetchWithTimeout(`${API_BASE_URL}/user_profile`, {
      method: 'PUT',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ profile })
    });
    
    const data = await response.json();
    
    if (!response.ok) {
      throw new Error(data.message || "Erreur lors de la mise à jour du profil utilisateur");
    }
    
    return data;
  } catch (error) {
    if (error instanceof Error) {
      throw error;
    }
    throw new Error("Erreur inattendue lors de la mise à jour du profil utilisateur");
  }
};

export default {
  checkHealth,
  initializeAssistant,
  sendMessage,
  getUserProfile,
  updateUserProfile
}; 