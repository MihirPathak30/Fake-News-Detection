document.addEventListener('DOMContentLoaded', () => {
    const form = document.getElementById('predictForm');
    const fileInput = document.getElementById('news_file');
    const fileName = document.getElementById('fileName');
    const resetButton = document.getElementById('resetButton');
    const loadingBox = document.getElementById('loadingBox');
    const messageBox = document.getElementById('messageBox');
    const resultCard = document.getElementById('resultCard');

    if (fileInput && fileName) {
        fileInput.addEventListener('change', () => {
            fileName.textContent = fileInput.files.length ? fileInput.files[0].name : 'No file selected';
        });
    }

    if (resetButton) {
        resetButton.addEventListener('click', () => {
            if (messageBox) {
                messageBox.textContent = '';
                messageBox.classList.add('hidden');
            }
            if (resultCard) {
                resultCard.classList.add('hidden');
            }
            if (fileName) {
                fileName.textContent = 'No file selected';
            }
        });
    }

    if (!form) {
        return;
    }

    const updateResult = (result) => {
        document.getElementById('resultLabel').textContent = result.label;
        document.getElementById('confidenceValue').textContent = `${(result.confidence * 100).toFixed(2)}%`;
        document.getElementById('timeValue').textContent = `${result.prediction_time_ms.toFixed(2)} ms`;
        document.getElementById('wordCountValue').textContent = result.word_count;
        document.getElementById('modelValue').textContent = result.model_used;
        document.getElementById('realProbabilityText').textContent = `${(result.real_probability * 100).toFixed(2)}%`;
        document.getElementById('fakeProbabilityText').textContent = `${(result.fake_probability * 100).toFixed(2)}%`;
        document.getElementById('realProbabilityBar').style.width = `${result.real_probability * 100}%`;
        document.getElementById('fakeProbabilityBar').style.width = `${result.fake_probability * 100}%`;
        resultCard.classList.remove('hidden');
    };

    const showMessage = (message) => {
        if (!messageBox) return;
        messageBox.textContent = message;
        messageBox.classList.remove('hidden');
    };

    form.addEventListener('submit', async (event) => {
        event.preventDefault();
        const apiUrl = form.dataset.apiUrl;
        const formData = new FormData(form);

        if (messageBox) {
            messageBox.textContent = '';
            messageBox.classList.add('hidden');
        }
        if (loadingBox) {
            loadingBox.classList.remove('hidden');
        }

        try {
            const response = await fetch(apiUrl, {
                method: 'POST',
                body: formData,
            });
            const payload = await response.json();
            if (!response.ok || !payload.success) {
                throw new Error(payload.message || 'Prediction failed.');
            }
            updateResult(payload.result);
        } catch (error) {
            showMessage(error.message || 'Prediction failed. Please try again.');
        } finally {
            if (loadingBox) {
                loadingBox.classList.add('hidden');
            }
        }
    });
});
