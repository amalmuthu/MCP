#!/usr/bin/env python3
"""
Coinglass Trading Analytics REST API
Provides trading analytics via simple REST endpoints for MCP Worker
"""

import os
import sys
import logging
from typing import Optional, Dict, Any
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import psycopg
from psycopg.rows import dict_row
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stderr)]
)

logger = logging.getLogger(__name__)

# Database connection
PG_DSN = os.environ.get("DATABASE_URL", "postgresql://postgres:jX8%23L7p%40qT9v%21sF2wY4z%24KbM@172.232.113.97:5432/coinglass")

def get_db_connection():
    """Get database connection"""
    return psycopg.connect(PG_DSN, autocommit=True)

# Create FastAPI app
app = FastAPI(title="Coinglass Trading Analytics API")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# CORE ANALYTICS FUNCTIONS (Keep your existing functions)
# =============================================================================

def analyze_funding_arbitrage(
    min_rate: float,
    timeframe: str,
    lookback_hours: int
) -> Dict[str, Any]:
    """Core function to analyze funding rate arbitrage opportunities."""
    with get_db_connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'binance_futures_funding_rate_history'
            AND table_name LIKE %s
            LIMIT 100
        """, (f'%_{timeframe}',))
        
        tables = [row['table_name'] for row in cur.fetchall()]
        opportunities = []
        
        for table in tables:
            try:
                symbol = table.replace(f'_{timeframe}', '')
                
                cur.execute(f"""
                    SELECT 
                        funding_rate,
                        time,
                        price
                    FROM binance_futures_funding_rate_history."{table}"
                    WHERE time >= NOW() - INTERVAL '{lookback_hours} hours'
                    ORDER BY time DESC
                    LIMIT 10
                """)
                
                rows = cur.fetchall()
                if not rows:
                    continue
                
                latest = rows[0]
                latest_rate = float(latest['funding_rate'] or 0)
                
                if latest_rate >= min_rate:
                    avg_rate = sum(float(r['funding_rate'] or 0) for r in rows) / len(rows)
                    
                    opportunities.append({
                        'symbol': symbol.upper(),
                        'current_funding_rate': round(latest_rate * 100, 4),
                        'average_funding_rate': round(avg_rate * 100, 4),
                        'current_price': float(latest['price'] or 0),
                        'last_update': str(latest['time']),
                        'annualized_rate': round(latest_rate * 365 * 3 * 100, 2),
                        'signal': 'SHORT' if latest_rate > 0.01 else 'MONITOR'
                    })
                    
            except Exception as e:
                logger.debug(f"Error processing {table}: {e}")
                continue
        
        opportunities.sort(key=lambda x: x['current_funding_rate'], reverse=True)
        
        return {
            'total_opportunities': len(opportunities),
            'min_rate_threshold': min_rate * 100,
            'timeframe': timeframe,
            'opportunities': opportunities[:20],
            'analysis_time': datetime.now().isoformat()
        }


def analyze_liquidation_zones(
    symbol: str,
    timeframe: str,
    periods: int
) -> Dict[str, Any]:
    """Core function to analyze liquidation zones."""
    clean_symbol = symbol.lower().replace('-', '').replace('_', '')
    table_name = f"{clean_symbol}_{timeframe}"
    
    with get_db_connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        cur.execute("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'binance_futures_liquidation_aggregated_history'
            AND table_name = %s
        """, (table_name,))
        
        if not cur.fetchone():
            return {
                'error': f'No liquidation data for {symbol}',
                'symbol': symbol.upper()
            }
        
        cur.execute(f"""
            SELECT 
                time,
                long_liquidation,
                short_liquidation,
                total_liquidation,
                price
            FROM binance_futures_liquidation_aggregated_history."{table_name}"
            ORDER BY time DESC
            LIMIT %s
        """, (periods,))
        
        rows = list(cur.fetchall())
        
        if not rows:
            return {
                'error': 'No recent liquidation data',
                'symbol': symbol.upper()
            }
        
        total_long_liq = sum(float(r['long_liquidation'] or 0) for r in rows)
        total_short_liq = sum(float(r['short_liquidation'] or 0) for r in rows)
        total_liq = total_long_liq + total_short_liq
        
        high_liq_periods = sorted(
            rows,
            key=lambda x: float(x['total_liquidation'] or 0),
            reverse=True
        )[:5]
        
        long_ratio = (total_long_liq / total_liq * 100) if total_liq > 0 else 0
        short_ratio = (total_short_liq / total_liq * 100) if total_liq > 0 else 0
        
        if long_ratio > 60:
            sentiment = 'BEARISH (Long liquidations dominant)'
            risk_level = 'HIGH'
        elif short_ratio > 60:
            sentiment = 'BULLISH (Short liquidations dominant)'
            risk_level = 'HIGH'
        else:
            sentiment = 'NEUTRAL'
            risk_level = 'MEDIUM'
        
        return {
            'symbol': symbol.upper(),
            'timeframe': timeframe,
            'periods_analyzed': len(rows),
            'total_liquidations': round(total_liq, 2),
            'long_liquidations': round(total_long_liq, 2),
            'short_liquidations': round(total_short_liq, 2),
            'long_percentage': round(long_ratio, 2),
            'short_percentage': round(short_ratio, 2),
            'market_sentiment': sentiment,
            'risk_level': risk_level,
            'current_price': float(rows[0]['price'] or 0),
            'high_liquidation_zones': [
                {
                    'time': str(row['time']),
                    'price': float(row['price'] or 0),
                    'total_liquidation': float(row['total_liquidation'] or 0),
                    'long_liquidation': float(row['long_liquidation'] or 0),
                    'short_liquidation': float(row['short_liquidation'] or 0)
                }
                for row in high_liq_periods
            ],
            'analysis_time': datetime.now().isoformat()
        }


# =============================================================================
# REST API ENDPOINTS (NEW)
# =============================================================================

# Request models
class FundingRequest(BaseModel):
    min_rate: float = 0.01
    timeframe: str = "1d"
    lookback_hours: int = 24

class LiquidationRequest(BaseModel):
    symbol: str
    timeframe: str = "1d"
    periods: int = 30

class ListSymbolsRequest(BaseModel):
    limit: int = 50

class SearchSymbolRequest(BaseModel):
    query: str


@app.post("/api/funding_arbitrage")
async def api_funding_arbitrage(req: FundingRequest):
    """REST endpoint for funding arbitrage analysis"""
    try:
        result = analyze_funding_arbitrage(req.min_rate, req.timeframe, req.lookback_hours)
        return result
    except Exception as e:
        logger.error(f"Error in funding_arbitrage: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/liquidation_zones")
async def api_liquidation_zones(req: LiquidationRequest):
    """REST endpoint for liquidation zones analysis"""
    try:
        result = analyze_liquidation_zones(req.symbol, req.timeframe, req.periods)
        return result
    except Exception as e:
        logger.error(f"Error in liquidation_zones: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/list_symbols")
async def api_list_symbols(req: ListSymbolsRequest):
    """REST endpoint to list available symbols"""
    try:
        with get_db_connection() as conn, conn.cursor() as cur:
            cur.execute("""
                SELECT DISTINCT REGEXP_REPLACE(table_name, '_[0-9a-z]+$', '') as symbol
                FROM information_schema.tables 
                WHERE table_schema = 'binance_futures_funding_rate_history'
                ORDER BY symbol
                LIMIT %s
            """, (req.limit,))
            
            symbols = [row[0].upper() for row in cur.fetchall()]
            
            return {
                "symbols": symbols,
                "count": len(symbols)
            }
    except Exception as e:
        logger.error(f"Error listing symbols: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/search_symbol")
async def api_search_symbol(req: SearchSymbolRequest):
    """REST endpoint to search symbols"""
    try:
        with get_db_connection() as conn, conn.cursor() as cur:
            cur.execute("""
                SELECT DISTINCT REGEXP_REPLACE(table_name, '_[0-9a-z]+$', '') as symbol
                FROM information_schema.tables 
                WHERE table_schema = 'binance_futures_funding_rate_history'
                AND table_name LIKE %s
                ORDER BY symbol
                LIMIT 20
            """, (f'%{req.query.lower()}%',))
            
            symbols = [row[0].upper() for row in cur.fetchall()]
            
            return {
                "symbols": symbols,
                "query": req.query,
                "count": len(symbols)
            }
    except Exception as e:
        logger.error(f"Error searching symbols: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Coinglass Trading Analytics API",
        "status": "running",
        "version": "1.0.0",
        "endpoints": {
            "funding_arbitrage": "/api/funding_arbitrage",
            "liquidation_zones": "/api/liquidation_zones",
            "list_symbols": "/api/list_symbols",
            "search_symbol": "/api/search_symbol"
        }
    }


@app.get("/health")
async def health():
    """Database health check"""
    try:
        with get_db_connection() as conn, conn.cursor() as cur:
            cur.execute("SELECT 1")
        return {"status": "healthy", "database": "connected"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


# =============================================================================
# SERVER STARTUP
# =============================================================================

def main():
    """Main function to start the REST API server"""
    try:
        logger.info("🚀 Starting Coinglass Trading Analytics REST API...")
        
        # Test database connection
        with get_db_connection() as conn, conn.cursor() as cur:
            cur.execute("SELECT 1")
            logger.info("✅ Database connection successful")
        
        port = int(os.environ.get("PORT", 10000))
        
        logger.info(f"📡 Starting REST API server on 0.0.0.0:{port}")
        logger.info("📌 API Documentation: /docs")
        
        # Run FastAPI with uvicorn
        uvicorn.run(app, host="0.0.0.0", port=port)
            
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
    except Exception as e:
        logger.error(f"❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()