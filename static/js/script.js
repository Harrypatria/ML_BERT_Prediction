document.addEventListener('DOMContentLoaded', function() {
    // Elements
    const descriptionInput = document.getElementById('description-input');
    const predictButton = document.getElementById('predict-button');
    const clearButton = document.getElementById('clear-button');
    const sampleDropdown = document.getElementById('sample-dropdown');
    const resultsContainer = document.getElementById('results-container');
    const loadingContainer = document.getElementById('loading-container');
    const errorContainer = document.getElementById('error-container');
    const errorMessage = document.getElementById('error-message');
    const predictionValue = document.getElementById('prediction-value');
    const confidenceValue = document.getElementById('confidence-value');
    const confidenceBadge = document.getElementById('confidence-badge');
    const analyzedText = document.getElementById('analyzed-text');
    
    // Chart
    let probabilityChart = null;
    
    // Sample descriptions
    const samples = {
        'als': "This clinical trial evaluates the safety and efficacy of an investigational therapy in people with Amyotrophic Lateral Sclerosis (ALS). ALS is a progressive neurodegenerative disease affecting motor neurons in the brain and spinal cord, leading to muscle weakness and eventual respiratory failure. The study measures changes in the ALSFRS-R score and survival time.",
        'dementia': "This study investigates a new treatment for patients with mild to moderate Alzheimer's disease dementia. The trial evaluates whether the drug can slow cognitive decline and improve daily functioning. Primary outcomes include changes in cognitive performance measured by ADAS-Cog and CDR-SB.",
        'ocd': "This trial examines a combined therapy approach for treatment-resistant Obsessive-Compulsive Disorder (OCD). OCD is characterized by unwanted thoughts and repetitive behaviors that impair functioning. The study measures symptom reduction using the Yale-Brown Obsessive Compulsive Scale.",
        'parkinsons': "This study evaluates a new dopamine agonist for early-stage Parkinson's disease. Parkinson's is characterized by tremor, rigidity, and slowness of movement due to dopamine cell loss. The trial measures changes in the Unified Parkinson's Disease Rating Scale scores.",
        'schizophrenia': "This clinical trial assesses the efficacy of a novel antipsychotic medication for treating schizophrenia symptoms. The study focuses on reducing positive symptoms like hallucinations and delusions, as well as negative symptoms like reduced emotional expression and avolition. Primary outcomes are measured using the PANSS scale over a 12-week treatment period."
    };
    
    // Medical terms mapping for highlighting
    const medicalTerms = {
        'ALS': ['ALS', 'amyotrophic lateral sclerosis', 'motor neuron', 'muscle weakness', 'respiratory failure', 'ALSFRS', 'bulbar', 'spinal', 'progressive', 'neurodegenerative'],
        'Dementia': ['dementia', 'Alzheimer', 'cognitive decline', 'memory', 'ADAS-Cog', 'CDR-SB', 'mild cognitive impairment', 'cognitive function'],
        'Obsessive Compulsive Disorder': ['OCD', 'obsessive', 'compulsive', 'ritual', 'intrusive thought', 'anxiety', 'Yale-Brown', 'repetitive'],
        'Parkinson\'s Disease': ['Parkinson', 'dopamine', 'tremor', 'rigidity', 'bradykinesia', 'movement disorder', 'UPDRS', 'motor symptom'],
        'Schizophrenia': ['schizophrenia', 'psychosis', 'hallucination', 'delusion', 'antipsychotic', 'PANSS', 'negative symptom', 'positive symptom']
    };
    
    // Check if API is available on load
    checkApiStatus();
    
    // Event listeners
    predictButton.addEventListener('click', predictCondition);
    clearButton.addEventListener('click', clearAll);
    sampleDropdown.addEventListener('change', loadSample);
    
    // Functions
    function checkApiStatus() {
        fetch('/api/health')
            .then(response => response.json())
            .then(data => {
                const modelStatus = document.getElementById('model-status');
                if (data.model_loaded) {
                    modelStatus.textContent = '✓ PubMedBERT model loaded';
                    modelStatus.style.color = '#4CAF50';
                } else {
                    modelStatus.textContent = '⚠ Model not loaded';
                    modelStatus.style.color = '#FFAB40';
                }
            })
            .catch(error => {
                console.error('API health check failed:', error);
                const modelStatus = document.getElementById('model-status');
                modelStatus.textContent = '✗ API not available';
                modelStatus.style.color = '#CF6679';
            });
    }
    
    function loadSample() {
        const sample = sampleDropdown.value;
        if (sample && samples[sample]) {
            descriptionInput.value = samples[sample];
        }
        sampleDropdown.value = '';
    }
    
    function predictCondition() {
        const description = descriptionInput.value.trim();
        if (!description) {
            showError('Please enter a clinical trial description');
            return;
        }
        
        showLoading();
        
        fetch('/api/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                description: description
            })
        })
        .then(response => {
            if (!response.ok) {
                throw new Error(`API error: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            hideLoading();
            displayResults(data, description);
        })
        .catch(error => {
            hideLoading();
            showError(`Error: ${error.message}`);
            console.error('Prediction error:', error);
        });
    }
    
    function displayResults(data, description) {
        // Show results container
        resultsContainer.classList.remove('hidden');
        errorContainer.classList.add('hidden');
        
        // Set prediction and confidence
        predictionValue.textContent = data.prediction || 'Unknown';
        
        const confidence = data.confidence || 0;
        confidenceValue.textContent = `${(confidence * 100).toFixed(1)}%`;
        
        // Set confidence badge
        setConfidenceBadge(confidence);
        
        // Highlight medical terms in text
        const highlightedText = highlightTerms(description, data.prediction);
        analyzedText.innerHTML = highlightedText;
        
        // Create/update chart
        window.probabilities = data.all_scores || {};
        createProbabilityChart(window.probabilities);
        
        // Scroll to results
        resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
    
    function setConfidenceBadge(confidence) {
        // Remove all classes first
        confidenceBadge.classList.remove('high', 'medium', 'low');
        
        // Set appropriate class and text
        if (confidence > 0.9) {
            confidenceBadge.textContent = 'Very High Confidence';
            confidenceBadge.classList.add('high');
        } else if (confidence > 0.7) {
            confidenceBadge.textContent = 'High Confidence';
            confidenceBadge.classList.add('high');
        } else if (confidence > 0.5) {
            confidenceBadge.textContent = 'Moderate Confidence';
            confidenceBadge.classList.add('medium');
        } else {
            confidenceBadge.textContent = 'Low Confidence';
            confidenceBadge.classList.add('low');
        }
    }
    
    function highlightTerms(text, condition) {
        if (!condition || !medicalTerms[condition]) {
            return text;
        }
        
        let highlightedText = text;
        const terms = medicalTerms[condition];
        
        terms.forEach(term => {
            const regex = new RegExp(`\\b${term}\\b`, 'gi');
            highlightedText = highlightedText.replace(regex, match => {
                return `<span class="key-term">${match}</span>`;
            });
        });
        
        return highlightedText;
    }
    
    function createProbabilityChart(probabilities) {
        // Clear previous chart
        const container = document.getElementById('probability-chart');
        container.innerHTML = '';
        
        // Sort data by probability
        const sortedData = Object.entries(probabilities)
            .sort((a, b) => b[1] - a[1]);
        
        // Chart dimensions
        const width = container.clientWidth;
        const height = 160;
        const margin = { top: 10, right: 60, bottom: 20, left: 140 };
        const innerWidth = width - margin.left - margin.right;
        const innerHeight = height - margin.top - margin.bottom;
        
        // Create SVG
        const svg = d3.select('#probability-chart')
            .append('svg')
            .attr('width', width)
            .attr('height', height)
            .append('g')
            .attr('transform', `translate(${margin.left},${margin.top})`);
        
        // Scales
        const xScale = d3.scaleLinear()
            .domain([0, 100])
            .range([0, innerWidth]);
        
        const yScale = d3.scaleBand()
            .domain(sortedData.map(d => d[0]))
            .range([0, innerHeight])
            .padding(0.3);
        
        // Add X axis
        svg.append('g')
            .attr('transform', `translate(0,${innerHeight})`)
            .call(d3.axisBottom(xScale)
                .ticks(5)
                .tickFormat(d => d + '%'))
            .call(g => g.select('.domain').remove())
            .call(g => g.selectAll('.tick line')
                .attr('stroke', 'rgba(255, 255, 255, 0.1)'))
            .call(g => g.selectAll('.tick text')
                .attr('fill', 'rgba(255, 255, 255, 0.7)')
                .attr('font-size', '11px'));
        
        // Add Y axis
        svg.append('g')
            .call(d3.axisLeft(yScale))
            .call(g => g.select('.domain').remove())
            .call(g => g.selectAll('.tick line').remove())
            .call(g => g.selectAll('.tick text')
                .attr('fill', 'rgba(255, 255, 255, 0.7)')
                .attr('font-size', '11px'));
        
        // Add lines
        svg.selectAll('line.lollipop-line')
            .data(sortedData)
            .enter()
            .append('line')
            .attr('class', 'lollipop-line')
            .attr('x1', 0)
            .attr('x2', d => xScale(d[1] * 100))
            .attr('y1', d => yScale(d[0]) + yScale.bandwidth() / 2)
            .attr('y2', d => yScale(d[0]) + yScale.bandwidth() / 2);
        
        // Add circles
        svg.selectAll('circle.lollipop-circle')
            .data(sortedData)
            .enter()
            .append('circle')
            .attr('class', 'lollipop-circle')
            .attr('cx', d => xScale(d[1] * 100))
            .attr('cy', d => yScale(d[0]) + yScale.bandwidth() / 2)
            .attr('r', 4);
        
        // Add percentage labels
        svg.selectAll('text.probability-label')
            .data(sortedData)
            .enter()
            .append('text')
            .attr('class', 'probability-label')
            .attr('x', d => xScale(d[1] * 100) + 5)
            .attr('y', d => yScale(d[0]) + yScale.bandwidth() / 2)
            .attr('dy', '0.35em')
            .text(d => `${(d[1] * 100).toFixed(1)}%`);
    }
    
    // Add window resize handler to make the chart responsive
    let resizeTimeout;
    window.addEventListener('resize', () => {
        clearTimeout(resizeTimeout);
        resizeTimeout = setTimeout(() => {
            const container = document.getElementById('probability-chart');
            if (container && container.innerHTML && window.probabilities) {
                createProbabilityChart(window.probabilities);
            }
        }, 250);
    });
    
    function clearAll() {
        descriptionInput.value = '';
        resultsContainer.classList.add('hidden');
        errorContainer.classList.add('hidden');
    }
    
    function showLoading() {
        loadingContainer.classList.remove('hidden');
        resultsContainer.classList.add('hidden');
        errorContainer.classList.add('hidden');
    }
    
    function hideLoading() {
        loadingContainer.classList.add('hidden');
    }
    
    function showError(message) {
        errorContainer.classList.remove('hidden');
        errorMessage.textContent = message;
    }
});
