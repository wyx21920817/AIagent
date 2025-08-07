import os
import requests
from fastapi import HTTPException
from typing import Optional

class AuthClient:
    def __init__(self):
        self.auth_url = os.getenv('AUTH_SERVER_URL', 'http://54.79.181.49:8001')
        self._token_cache = None
    
    def get_temp_token(self, force_new: bool = False) -> str:
        """Get a temporary token from the authentication server.
        
        Args:
            force_new: If True, ignore cached token and get a new one
            
        Returns:
            str: The temporary token
            
        Raises:
            HTTPException: If unable to get token from auth server
        """
        # Return cached token if available and not forcing new
        if not force_new and self._token_cache:
            return self._token_cache
            
        headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        }
        
        try:
            response = requests.get(
                f'{self.auth_url}/api/auth/temp-token',
                headers=headers,
                verify=False,
                timeout=5.0
            )
            response.raise_for_status()
            data = response.json()
            
            temp_token = data.get('token')
            if not temp_token:
                raise ValueError("Token not found in response")
                
            # Cache the token
            self._token_cache = temp_token
            return temp_token
            
        except requests.Timeout:
            raise HTTPException(
                status_code=504,
                detail="Token request timed out. Please try again later."
            )
        except requests.HTTPError as e:
            raise HTTPException(
                status_code=e.response.status_code,
                detail=f"Failed to get token: {e.response.text}"
            )
        except requests.RequestException as e:
            raise HTTPException(
                status_code=503,
                detail="Unable to connect to authentication server. Please check network connection."
            )

# Create a singleton instance
_auth_client = None

def get_auth_client() -> AuthClient:
    """Get the singleton AuthClient instance."""
    global _auth_client
    if _auth_client is None:
        _auth_client = AuthClient()
    return _auth_client

def get_temp_token(force_new: bool = False) -> str:
    """Convenience function to get a temporary token."""
    client = get_auth_client()
    return client.get_temp_token(force_new)
