import React, { useState } from 'react';

const StockPrediction = () => {
    const [ticker, setTicker] = useState('');
    const [time, setTime] = useState('');
    const [prediction, setPrediction] = useState(null);
    const [error, setError] = useState(null);

    const handleSubmit = async (e) => {
        e.preventDefault();
        setError(null); // Reset error state

        try {
            const response = await fetch('http://localhost:5000/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ ticker, time: parseInt(time) })
            });

            const result = await response.json();
            if (response.ok) {
                setPrediction(result.predictions);
            } else {
                setError(result.error);
            }
        } catch (err) {
            setError('An unexpected error occurred.');
        }
    };

    return (
        <div>
            <h1>Stock Prediction</h1>
            <form onSubmit={handleSubmit}>
                <div>
                    <label>
                        Ticker:
                        <input 
                            type="text" 
                            value={ticker} 
                            onChange={(e) => setTicker(e.target.value)} 
                            required 
                        />
                    </label>
                </div>
                <div>
                    <label>
                        Time (days):
                        <input 
                            type="number" 
                            value={time} 
                            onChange={(e) => setTime(e.target.value)} 
                            required 
                        />
                    </label>
                </div>
                <button type="submit">Get Prediction</button>
            </form>

            {prediction && (
                <div>
                    <h2>Prediction</h2>
                    <p>{prediction}</p>
                </div>
            )}

            {error && (
                <div>
                    <h2>Error</h2>
                    <p>{error}</p>
                </div>
            )}
        </div>
    );
};

export default StockPrediction;
