// Age Classifier Modal Logic
document.addEventListener('DOMContentLoaded', function () {
    const ageClassifierBtn = document.getElementById('age-classifier-btn');
    const ageModal = document.getElementById('age-modal');
    const closeModal = document.querySelector('.close-modal');
    const classifyBtn = document.getElementById('classify-btn');
    const ageInput = document.getElementById('age-input');
    const ageUnit = document.getElementById('age-unit');
    const classifierResult = document.getElementById('classifier-result');
    const classifierMessage = document.getElementById('classifier-message');
    const classifierLink = document.getElementById('classifier-link');

    // Open modal
    if (ageClassifierBtn) {
        ageClassifierBtn.addEventListener('click', function () {
            ageModal.style.display = 'flex';
            ageInput.value = '';
            classifierResult.classList.add('hidden');
        });
    }

    // Close modal
    if (closeModal) {
        closeModal.addEventListener('click', function () {
            ageModal.style.display = 'none';
        });
    }

    // Close modal when clicking outside
    window.addEventListener('click', function (event) {
        if (event.target === ageModal) {
            ageModal.style.display = 'none';
        }
    });

    // Classify age
    if (classifyBtn) {
        classifyBtn.addEventListener('click', async function () {
            const age = parseFloat(ageInput.value);
            const unit = ageUnit.value;

            if (!age || age <= 0) {
                alert('Please enter a valid age.');
                return;
            }

            try {
                const response = await fetch('/classify_age', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ age: age, unit: unit })
                });

                const data = await response.json();

                if (data.error) {
                    alert(data.error);
                } else {
                    classifierMessage.textContent = data.message;
                    classifierLink.href = data.redirect_url;
                    classifierResult.classList.remove('hidden');
                }
            } catch (error) {
                console.error('Error:', error);
                alert('An error occurred. Please try again.');
            }
        });
    }

    // Form validation
    const screeningForm = document.getElementById('screeningForm');
    if (screeningForm) {
        screeningForm.addEventListener('submit', function (event) {
            // Check if all A1-A10 questions are answered
            let allAnswered = true;
            for (let i = 1; i <= 10; i++) {
                const radios = document.getElementsByName(`A${i}_Score`);
                let answered = false;
                for (let radio of radios) {
                    if (radio.checked) {
                        answered = true;
                        break;
                    }
                }
                if (!answered) {
                    allAnswered = false;
                    break;
                }
            }

            if (!allAnswered) {
                event.preventDefault();
                alert('Please answer all behavioral questions (A1-A10).');
                return false;
            }

            // Additional validation can be added here
            return true;
        });
    }
});
