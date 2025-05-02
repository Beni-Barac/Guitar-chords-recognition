document.addEventListener('DOMContentLoaded', function() {
    // DOM elements
    const uploadForm = document.getElementById('upload-form');
    const audioFileInput = document.getElementById('audio-file');
    const recordBtn = document.getElementById('record-btn');
    const stopBtn = document.getElementById('stop-btn');
    const recordingTimer = document.getElementById('recording-timer');
    const resultsSection = document.getElementById('results-section');
    const audioPlayerSection = document.getElementById('audio-player-section');
    const audioPlayer = document.getElementById('audio-player');
    const spectrogramImg = document.getElementById('spectrogram');
    const chordNameElement = document.getElementById('chord-name');
    const confidenceElement = document.getElementById('confidence');
    const loadingElement = document.getElementById('loading');

    // Recording variables
    let mediaRecorder;
    let audioChunks = [];
    let startTime;
    let timerInterval;

    // Handle file upload form submission
    uploadForm.addEventListener('submit', function(e) {
        e.preventDefault();

        const file = audioFileInput.files[0];
        if (!file) {
            alert('Please select an audio file.');
            return;
        }

        // Show loading indicator
        loadingElement.classList.remove('hidden');

        // Create form data and send to the server
        const formData = new FormData();
        formData.append('file', file);

        // Send to server for analysis
        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                alert('Error: ' + data.error);
                return;
            }

            // Update UI with results
            showResults(data);
        })
        .catch(error => {
            console.error('Error:', error);
            alert('An error occurred during analysis.');
        })
        .finally(() => {
            // Hide loading indicator
            loadingElement.classList.add('hidden');
        });
    });

    // Recording functionality
    recordBtn.addEventListener('click', async function() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

            // Set up media recorder
            mediaRecorder = new MediaRecorder(stream);
            audioChunks = [];

            // Collect audio chunks as they become available
            mediaRecorder.addEventListener('dataavailable', e => {
                audioChunks.push(e.data);
            });

            // When recording stops
            mediaRecorder.addEventListener('stop', async () => {
                // Stop the timer
                clearInterval(timerInterval);

                // Create audio blob and URL
                const audioBlob = new Blob(audioChunks, { type: 'audio/wav' });
                const audioUrl = URL.createObjectURL(audioBlob);

                // Update audio player
                audioPlayer.src = audioUrl;
                audioPlayerSection.classList.remove('hidden');

                // Show loading indicator
                loadingElement.classList.remove('hidden');

                // Submit recorded audio for analysis
                const formData = new FormData();
                formData.append('file', audioBlob, 'recorded_audio.wav');

                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        body: formData
                    });

                    const data = await response.json();
                    if (data.error) {
                        alert('Error: ' + data.error);
                        return;
                    }

                    // Update UI with results
                    showResults(data);
                } catch (error) {
                    console.error('Error:', error);
                    alert('An error occurred during analysis.');
                } finally {
                    // Hide loading indicator
                    loadingElement.classList.add('hidden');
                }

                // Stop all tracks
                stream.getTracks().forEach(track => track.stop());
            });

            // Start recording
            mediaRecorder.start();

            // Update UI
            recordBtn.disabled = true;
            stopBtn.disabled = false;

            // Start timer
            startTime = Date.now();
            updateTimer();
            timerInterval = setInterval(updateTimer, 1000);

        } catch (err) {
            console.error('Error accessing microphone:', err);
            alert('Could not access microphone. Please check permissions.');
        }
    });

    // Stop recording button
    stopBtn.addEventListener('click', function() {
        if (mediaRecorder && mediaRecorder.state !== 'inactive') {
            mediaRecorder.stop();
            recordBtn.disabled = false;
            stopBtn.disabled = true;
        }
    });

    // Update recording timer display
    function updateTimer() {
        const elapsedTime = Math.floor((Date.now() - startTime) / 1000);
        const minutes = Math.floor(elapsedTime / 60).toString().padStart(2, '0');
        const seconds = (elapsedTime % 60).toString().padStart(2, '0');
        recordingTimer.textContent = `${minutes}:${seconds}`;
    }

    // Show analysis results in the UI
    function showResults(data) {
        // Display spectrogram
        spectrogramImg.src = data.spectrogram;

        // Display chord prediction
        chordNameElement.textContent = data.chord;
        confidenceElement.textContent = `Confidence: ${data.confidence}`;

        // Show results section
        resultsSection.classList.remove('hidden');
    }
});