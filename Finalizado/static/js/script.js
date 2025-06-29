// Variáveis globais
let teams = [];
let rounds = [];

// Elementos DOM
const manданteSelect = document.getElementById('mandante');
const visitanteSelect = document.getElementById('visitante');
const rodadaSelect = document.getElementById('rodada');
const predictionForm = document.getElementById('prediction-form');
const resultsCard = document.getElementById('results-card');
const loading = document.getElementById('loading');
const matchInfo = document.getElementById('match-info');
const predictionsContainer = document.getElementById('predictions-container');

// Inicialização
document.addEventListener('DOMContentLoaded', function() {
    loadTeams();
    loadRounds();
    setupEventListeners();
});

// Carrega lista de times
async function loadTeams() {
    try {
        const response = await fetch('/api/times');
        const data = await response.json();
        
        if (data.error) {
            throw new Error(data.error);
        }
        
        teams = data.times;
        populateTeamSelects();
    } catch (error) {
        console.error('Erro ao carregar times:', error);
        showError('Erro ao carregar lista de times');
    }
}

// Carrega lista de rodadas
async function loadRounds() {
    try {
        const response = await fetch('/api/rodadas');
        const data = await response.json();
        
        if (data.error) {
            throw new Error(data.error);
        }
        
        rounds = data.rodadas;
        populateRoundSelect();
    } catch (error) {
        console.error('Erro ao carregar rodadas:', error);
        showError('Erro ao carregar lista de rodadas');
    }
}

// Popula selects de times
function populateTeamSelects() {
    // Limpa selects
    manданteSelect.innerHTML = '<option value="">Selecione o time da casa</option>';
    visitanteSelect.innerHTML = '<option value="">Selecione o time visitante</option>';
    
    // Adiciona opções
    teams.forEach(team => {
        const option1 = new Option(team, team);
        const option2 = new Option(team, team);
        manданteSelect.add(option1);
        visitanteSelect.add(option2);
    });
}

// Popula select de rodadas
function populateRoundSelect() {
    rodadaSelect.innerHTML = '<option value="">Selecione a rodada</option>';
    
    rounds.forEach(round => {
        const option = new Option(`Rodada ${round}`, round);
        rodadaSelect.add(option);
    });
    
    // Seleciona a rodada 1 por padrão
    if (rounds.length > 0) {
        // Se a rodada 1 existe nos dados, seleciona ela
        if (rounds.includes(1)) {
            rodadaSelect.value = 1;
        } else {
            // Caso contrário, seleciona a primeira rodada disponível
            rodadaSelect.value = Math.min(...rounds);
        }
    }
}

// Configura event listeners
function setupEventListeners() {
    predictionForm.addEventListener('submit', handleFormSubmit);
    
    // Evita que o mesmo time seja selecionado nos dois selects
    manданteSelect.addEventListener('change', function() {
        updateTeamOptions(this.value, visitanteSelect);
    });
    
    visitanteSelect.addEventListener('change', function() {
        updateTeamOptions(this.value, manданteSelect);
    });
}

// Atualiza opções de times para evitar duplicação
function updateTeamOptions(selectedTeam, otherSelect) {
    const currentValue = otherSelect.value;
    
    // Remove todas as opções exceto a primeira
    otherSelect.innerHTML = otherSelect.options[0].outerHTML;
    
    // Adiciona todas as opções exceto o time selecionado
    teams.forEach(team => {
        if (team !== selectedTeam) {
            const option = new Option(team, team);
            otherSelect.add(option);
        }
    });
    
    // Restaura valor selecionado se ainda for válido
    if (currentValue && currentValue !== selectedTeam) {
        otherSelect.value = currentValue;
    }
}

// Manipula envio do formulário
async function handleFormSubmit(event) {
    event.preventDefault();
    
    const formData = new FormData(predictionForm);
    const data = {
        time_mandante: formData.get('mandante'),
        time_visitante: formData.get('visitante'),
        rodada: parseInt(formData.get('rodada'))
    };
    
    // Validação
    if (!data.time_mandante || !data.time_visitante || !data.rodada) {
        showError('Por favor, preencha todos os campos');
        return;
    }
    
    if (data.time_mandante === data.time_visitante) {
        showError('Times mandante e visitante devem ser diferentes');
        return;
    }
    
    await makePrediction(data);
}

// Faz a predição
async function makePrediction(data) {
    showLoading(true);
    hideResults();
    
    try {
        const response = await fetch('/api/prever', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data)
        });
        
        const result = await response.json();
        
        if (result.error) {
            throw new Error(result.error);
        }
        
        displayResults(result);
    } catch (error) {
        console.error('Erro na predição:', error);
        showError('Erro ao fazer predição: ' + error.message);
    } finally {
        showLoading(false);
    }
}

// Exibe resultados
function displayResults(result) {
    const { time_mandante, time_visitante, rodada, probabilidades } = result;
    
    // Atualiza informações da partida
    matchInfo.innerHTML = `
        <div class="match-title">
            ${time_mandante} <span style="color: #dc3545;">VS</span> ${time_visitante}
        </div>
        <div class="match-details">
            Rodada ${rodada} - Análise Preditiva
        </div>
    `;
    
    // Ordena probabilidades por valor
    const sortedProbs = Object.entries(probabilidades)
        .sort(([,a], [,b]) => b - a);
    
    // Gera cards de predição
    const predictionCards = sortedProbs.map(([tipo, prob]) => {
        const percentage = (prob * 100).toFixed(1);
        const icon = getResultIcon(tipo);
        const label = getResultLabel(tipo, time_mandante, time_visitante);
        
        return `
            <div class="prediction-item ${tipo}">
                <div class="prediction-icon">${icon}</div>
                <div class="prediction-label">${label}</div>
                <div class="prediction-percentage">${percentage}%</div>
                <div class="prediction-bar">
                    <div class="prediction-fill" style="width: ${percentage}%"></div>
                </div>
            </div>
        `;
    }).join('');
    
    predictionsContainer.innerHTML = predictionCards;
    
    // Anima a exibição dos resultados
    setTimeout(() => {
        resultsCard.style.display = 'block';
        resultsCard.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }, 300);
}

// Retorna ícone para cada tipo de resultado
function getResultIcon(tipo) {
    const icons = {
        'vitoria': '<i class="fas fa-trophy"></i>',
        'empate': '<i class="fas fa-handshake"></i>',
        'derrota': '<i class="fas fa-futbol"></i>'
    };
    return icons[tipo] || '<i class="fas fa-question"></i>';
}

// Retorna label para cada tipo de resultado
function getResultLabel(tipo, mandante, visitante) {
    const labels = {
        'vitoria': `Vitória ${mandante}`,
        'empate': 'Empate',
        'derrota': `Vitória ${visitante}`
    };
    return labels[tipo] || 'Resultado Desconhecido';
}

// Exibe/oculta loading
function showLoading(show) {
    loading.style.display = show ? 'block' : 'none';
}

// Oculta resultados
function hideResults() {
    resultsCard.style.display = 'none';
}

// Exibe erro
function showError(message) {
    // Remove alertas anteriores
    const existingAlert = document.querySelector('.error-alert');
    if (existingAlert) {
        existingAlert.remove();
    }
    
    // Cria novo alerta
    const alert = document.createElement('div');
    alert.className = 'error-alert';
    alert.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: linear-gradient(135deg, #dc3545, #c82333);
        color: white;
        padding: 15px 20px;
        border-radius: 10px;
        box-shadow: 0 5px 15px rgba(220, 53, 69, 0.3);
        z-index: 1000;
        max-width: 400px;
        font-weight: 500;
        animation: slideIn 0.3s ease;
    `;
    alert.innerHTML = `
        <div style="display: flex; align-items: center; gap: 10px;">
            <i class="fas fa-exclamation-triangle"></i>
            <span>${message}</span>
            <button onclick="this.parentElement.parentElement.remove()" 
                    style="background: none; border: none; color: white; font-size: 1.2rem; cursor: pointer; margin-left: auto;">
                <i class="fas fa-times"></i>
            </button>
        </div>
    `;
    
    document.body.appendChild(alert);
    
    // Remove automaticamente após 5 segundos
    setTimeout(() => {
        if (alert.parentElement) {
            alert.remove();
        }
    }, 5000);
}

// Adiciona animação CSS via JavaScript
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
`;
document.head.appendChild(style);
