import React, { useState } from 'react';
import axios from 'axios';

function App() {
  const [textInput, setTextInput] = useState('');
  const [fileInput, setFileInput] = useState(null);
  const [audioUrl, setAudioUrl] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  // Replace this with your actual API endpoint
  const API_ENDPOINT = 'http://127.0.0.1:5000/process-text';

  const handleTextChange = (e) => {
    setTextInput(e.target.value);
  };

  const handleFileChange = (e) => {
    setFileInput(e.target.files[0]);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    let content = textInput;

    // If a file is selected, read its content
    if (fileInput) {
      const reader = new FileReader();
      reader.onload = async (event) => {
        content = event.target.result;
        await sendData(content);
      };
      reader.readAsText(fileInput);
    } else if (textInput.trim() !== '') {
      await sendData(content);
    } else {
      alert('Please enter text or select a text file.');
    }
  };

  const sendData = async (content) => {
    setIsLoading(true);
    try {
      const response = await axios.post(API_ENDPOINT, {
        text: content,
      });

      if (response.status === 200) {
        setAudioUrl(response.data.audioUrl);
      } else {
        alert('Error processing your request.');
      }
    } catch (error) {
      console.error('Error:', error);
      alert('An error occurred while processing your request.');
    }
    setIsLoading(false);
  };

  return (
    <div style={styles.container}>
      <h1>Text-to-Speech Uploader</h1>
      <form onSubmit={handleSubmit} style={styles.form}>
        <textarea
          placeholder="Enter text here..."
          value={textInput}
          onChange={handleTextChange}
          rows={10}
          style={styles.textarea}
        />
        <p>Or</p>
        <input type="file" accept=".txt" onChange={handleFileChange} />
        <button type="submit" style={styles.button} disabled={isLoading}>
          {isLoading ? 'Processing...' : 'Submit'}
        </button>
      </form>
      {audioUrl && (
        <div style={styles.audioContainer}>
          <h2>Generated Audio:</h2>
          <audio controls src={audioUrl} style={styles.audioPlayer} />
          <p>
            <a href={audioUrl} target="_blank" rel="noopener noreferrer">
              Download Audio
            </a>
          </p>
        </div>
      )}
    </div>
  );
}

const styles = {
  container: {
    fontFamily: 'Arial, sans-serif',
    maxWidth: 600,
    margin: '0 auto',
    padding: 20,
    textAlign: 'center',
  },
  form: {
    marginBottom: 20,
  },
  textarea: {
    width: '100%',
    padding: 10,
    fontSize: 16,
  },
  button: {
    padding: '10px 20px',
    fontSize: 16,
    marginTop: 10,
  },
  audioContainer: {
    marginTop: 20,
  },
  audioPlayer: {
    width: '100%',
  },
};

export default App;
