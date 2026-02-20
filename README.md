# Quant Portfolio Optimizer 📈

## Opis projektu
Narzędzie do automatycznej optymalizacji portfela inwestycyjnego oparte na **Modern Portfolio Theory**. Skrypt pobiera dane historyczne spółek i oblicza optymalne wagi aktywów, aby zmaksymalizować zysk przy minimalnym ryzyku.

## Kluczowe Funkcjonalności
* **Automatyczne pobieranie danych:** Wykorzystanie API `yfinance`.
* **Optymalizacja Markowitza:** Wyznaczanie portfela o maksymalnym wskaźniku Sharpe'a.
* **Analiza ryzyka:** Obliczanie macierzy kowariancji i oczekiwanych stóp zwrotu.

## Zastosowana Matematyka
Głównym celem modelu jest znalezienie wag portfela, które maksymalizują **Wskaźnik Sharpe'a**:

## Technologia
* **Język:** Python 3.x
* **Biblioteki:** `PyPortfolioOpt`, `pandas`, `yfinance`
