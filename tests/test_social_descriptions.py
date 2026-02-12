import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Mock dependencies that might fail during import in test environment
sys.modules['faster_whisper'] = MagicMock()
sys.modules['moviepy'] = MagicMock()
sys.modules['pydub'] = MagicMock()
sys.modules['nltk'] = MagicMock()
sys.modules['openai'] = MagicMock()

from skills.transcription.tool import AudioTranscriberSummarizer

class TestSocialDescriptions(unittest.TestCase):
    def setUp(self):
        # Patch WhisperModel initialization to avoid actual model loading
        with patch('faster_whisper.WhisperModel'):
            self.ats = AudioTranscriberSummarizer(model_size="base")
            self.ats.hf_token = "fake_token"

    @patch('openai.OpenAI')
    def test_generate_social_descriptions_parsing(self, mock_openai_class):
        # Mock client instance
        mock_client = mock_openai_class.return_value
        
        # Mock response structure: response.choices[0].message.content
        mock_response = MagicMock()
        mock_message = MagicMock()
        mock_message.content = (
            "Opción 1 (Filosofía):\n"
            "El éxito no se mide por un gran negocio, sino por diez pequeños pasos bien dados.\n\n"
            "Opción 2 (Lección):\n"
            "Aprendan de Carlos Slim: la ingeniería y las finanzas van de la mano. Los balances son la clave.\n\n"
            "Opción 3 (Aprendizaje):\n"
            "Descubre cómo un estudiante de ingeniería llegó a dominar el mundo de las inversiones."
        )
        mock_response.choices = [MagicMock(message=mock_message)]
        
        # Mock completions.create call
        mock_client.chat.completions.create.return_value = mock_response
        
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
