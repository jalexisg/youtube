import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Mock dependencies that might fail during import in test environment
sys.modules['faster_whisper'] = MagicMock()
sys.modules['moviepy'] = MagicMock()
sys.modules['pydub'] = MagicMock()
sys.modules['nltk'] = MagicMock()
sys.modules['huggingface_hub'] = MagicMock()

from skills.transcription.tool import AudioTranscriberSummarizer

class TestSocialDescriptions(unittest.TestCase):
    def setUp(self):
        # Patch WhisperModel initialization to avoid actual model loading
        with patch('faster_whisper.WhisperModel'):
            self.ats = AudioTranscriberSummarizer(model_size="base")
            self.ats.hf_token = "fake_token"

    @patch('huggingface_hub.InferenceClient')
    def test_generate_social_descriptions_parsing(self, mock_client_class):
        # Mock client response
        mock_client = mock_client_class.return_value
        
        # Simulate a typical response from the model
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].delta.content = (
            "Opción 1 (Enfocada en su filosofía):\n"
            "El éxito no se mide por un gran negocio, sino por diez pequeños pasos bien dados.\n\n"
            "Opción 2 (Presentada como una lección):\n"
            "Aprendan de Carlos Slim: la ingeniería y las finanzas van de la mano. Los balances son la clave.\n\n"
            "Opción 3 (Enfocada en el aprendizaje):\n"
            "Descubre cómo un estudiante de ingeniería llegó a dominar el mundo de las inversiones."
        )
        
        # Mock chat_completion generator
        mock_client.chat_completion.return_value = [mock_response]
        
        # Call the method
        result = self.ats.generate_social_descriptions("Texto de prueba sobre negocios y Slim.")
        
        # Verify results
        self.assertIn("filosofia", result)
        self.assertIn("leccion", result)
        self.assertIn("aprendizaje", result)
        self.assertEqual(result["filosofia"], "El éxito no se mide por un gran negocio, sino por diez pequeños pasos bien dados.")
        self.assertEqual(result["leccion"], "Aprendan de Carlos Slim: la ingeniería y las finanzas van de la mano. Los balances son la clave.")
        self.assertEqual(result["aprendizaje"], "Descubre cómo un estudiante de ingeniería llegó a dominar el mundo de las inversiones.")

    def test_missing_token(self):
        self.ats.hf_token = None
        result = self.ats.generate_social_descriptions("Texto")
        self.assertEqual(result, {})

if __name__ == '__main__':
    unittest.main()
