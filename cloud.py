import psycopg
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
import uvicorn

app = FastAPI(
    title="CoinGlass Funding Rates API",
    description="API for cryptocurrency funding rates data",
    version="1.0.0"
)

# Add CORS for Cloudflare Workers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Database connection
PG_DSN = "postgresql://postgres:jX8%23L7p%40qT9v%21sF2wY4z%24KbM@172.232.113.97:5432/coinglass"

def get_db_connection():
    return psycopg.connect(PG_DSN, autocommit=True)

# Request models
class CompareRequest(BaseModel):
    symbols: List[str]
    days: int = 3

class QueryRequest(BaseModel):
    query: str

# ========================================
# ENDPOINTS
# ========================================

@app.get("/")
def home():
    """API Home - Service status"""
    return {
        "status": "running",
        "service": "CoinGlass Funding Rates API",
        "version": "1.0.0",
        "endpoints": {
            "GET /": "This page",
            "GET /pairs": "List all available crypto pairs",
            "GET /funding/{symbol}": "Get funding rate for a symbol",
            "GET /highest": "Get highest funding rates",
            "GET /lowest": "Get lowest funding rates",
            "GET /search/{term}": "Search for crypto symbols",
            "GET /stats/{symbol}": "Get statistics for a symbol",
            "POST /compare": "Compare multiple symbols",
            "POST /query": "Execute custom SQL query"
        }
    }

@app.get("/pairs")
def list_pairs(limit: int = 50):
    """
    Get list of available cryptocurrency pairs
    
    - **limit**: Maximum number of pairs to return (default: 50)
    """
    try:
        with get_db_connection() as conn, conn.cursor() as cur:
            cur.execute("""
                SELECT DISTINCT REPLACE(table_name, '_1d', '') as symbol
                FROM information_schema.tables 
                WHERE table_schema = 'funding_rate' 
                AND table_name LIKE '%%_1d'
                ORDER BY symbol
                LIMIT %s
            """, (limit,))
            pairs = [row[0] for row in cur.fetchall()]
            return {
                "success": True,
                "count": len(pairs),
                "pairs": pairs
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/funding/{symbol}")
def get_funding(symbol: str, days: int = 7):
    """
    Get funding rate data for a specific cryptocurrency pair
    
    - **symbol**: Crypto pair symbol (e.g., 'btcusdt', 'ethusdt')
    - **days**: Number of days of data (default: 7)
    """
    try:
        clean_symbol = symbol.lower().replace('-', '').replace('_', '')
        table_name = f"{clean_symbol}_1d"
        
        with get_db_connection() as conn, conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
            # Check if table exists
            cur.execute("""
                SELECT table_name FROM information_schema.tables 
                WHERE table_schema = 'funding_rate' AND table_name = %s
            """, (table_name,))
            
            if not cur.fetchone():
                raise HTTPException(
                    status_code=404, 
                    detail=f"No data found for symbol: {symbol}"
                )
            
            # Get funding rate data
            cur.execute(f"""
                SELECT * FROM funding_rate."{table_name}"
                WHERE time >= NOW() - INTERVAL '{days} days'
                ORDER BY time DESC
                LIMIT 100
            """)
            
            data = list(cur.fetchall())
            
            return {
                "success": True,
                "symbol": symbol,
                "days": days,
                "count": len(data),
                "data": data
            }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/highest")
def highest_rates(days: int = 1, limit: int = 10):
    """
    Get cryptocurrency pairs with highest funding rates
    
    - **days**: Number of days to look back (default: 1)
    - **limit**: Number of results to return (default: 10)
    """
    try:
        with get_db_connection() as conn, conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
            cur.execute("""
                SELECT table_name FROM information_schema.tables 
                WHERE table_schema = 'funding_rate' 
                AND table_name LIKE '%%_1d'
                LIMIT 50
            """)
            
            tables = [row['table_name'] for row in cur.fetchall()]
            results = []
            
            for table in tables:
                try:
                    cur.execute(f"""
                        SELECT funding_rate, time 
                        FROM funding_rate."{table}"
                        WHERE time >= NOW() - INTERVAL '{days} days'
                        ORDER BY time DESC
                        LIMIT 1
                    """)
                    
                    row = cur.fetchone()
                    if row and row['funding_rate']:
                        results.append({
                            'symbol': table.replace('_1d', ''),
                            'funding_rate': float(row['funding_rate']),
                            'time': str(row['time'])
                        })
                except:
                    continue
            
            results.sort(key=lambda x: x['funding_rate'], reverse=True)
            
            return {
                "success": True,
                "days": days,
                "count": len(results[:limit]),
                "highest": results[:limit]
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/lowest")
def lowest_rates(days: int = 1, limit: int = 10):
    """
    Get cryptocurrency pairs with lowest funding rates
    
    - **days**: Number of days to look back (default: 1)
    - **limit**: Number of results to return (default: 10)
    """
    try:
        with get_db_connection() as conn, conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
            cur.execute("""
                SELECT table_name FROM information_schema.tables 
                WHERE table_schema = 'funding_rate' 
                AND table_name LIKE '%%_1d'
                LIMIT 50
            """)
            
            tables = [row['table_name'] for row in cur.fetchall()]
            results = []
            
            for table in tables:
                try:
                    cur.execute(f"""
                        SELECT funding_rate, time 
                        FROM funding_rate."{table}"
                        WHERE time >= NOW() - INTERVAL '{days} days'
                        ORDER BY time DESC
                        LIMIT 1
                    """)
                    
                    row = cur.fetchone()
                    if row and row['funding_rate']:
                        results.append({
                            'symbol': table.replace('_1d', ''),
                            'funding_rate': float(row['funding_rate']),
                            'time': str(row['time'])
                        })
                except:
                    continue
            
            results.sort(key=lambda x: x['funding_rate'])
            
            return {
                "success": True,
                "days": days,
                "count": len(results[:limit]),
                "lowest": results[:limit]
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/search/{term}")
def search_symbol(term: str, limit: int = 20):
    """
    Search for cryptocurrency symbols
    
    - **term**: Search term (e.g., 'btc', 'eth')
    - **limit**: Maximum results (default: 20)
    """
    try:
        with get_db_connection() as conn, conn.cursor() as cur:
            cur.execute("""
                SELECT DISTINCT REPLACE(table_name, '_1d', '') as symbol
                FROM information_schema.tables 
                WHERE table_schema = 'funding_rate' 
                AND table_name LIKE '%%_1d'
                AND table_name LIKE %s
                ORDER BY symbol
                LIMIT %s
            """, (f"%{term.lower()}%", limit))
            
            matches = [row[0] for row in cur.fetchall()]
            
            return {
                "success": True,
                "search_term": term,
                "count": len(matches),
                "matches": matches
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats/{symbol}")
def get_statistics(symbol: str, days: int = 30):
    """
    Get statistical analysis for a cryptocurrency pair
    
    - **symbol**: Crypto pair symbol
    - **days**: Number of days to analyze (default: 30)
    """
    try:
        clean_symbol = symbol.lower().replace('-', '').replace('_', '')
        table_name = f"{clean_symbol}_1d"
        
        with get_db_connection() as conn, conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
            cur.execute("""
                SELECT table_name FROM information_schema.tables 
                WHERE table_schema = 'funding_rate' AND table_name = %s
            """, (table_name,))
            
            if not cur.fetchone():
                raise HTTPException(
                    status_code=404,
                    detail=f"No data found for symbol: {symbol}"
                )
            
            cur.execute(f"""
                SELECT * FROM funding_rate."{table_name}"
                WHERE time >= NOW() - INTERVAL '%s days'
                ORDER BY time DESC
            """, (days,))
            
            data = list(cur.fetchall())
            
            if not data:
                raise HTTPException(
                    status_code=404,
                    detail=f"No data available for the last {days} days"
                )
            
            rates = []
            for row in data:
                if row.get('funding_rate') is not None:
                    try:
                        rates.append(float(row['funding_rate']))
                    except:
                        continue
            
            if not rates:
                raise HTTPException(
                    status_code=404,
                    detail="No valid funding rate data found"
                )
            
            return {
                "success": True,
                "symbol": symbol,
                "period_days": days,
                "statistics": {
                    "data_points": len(rates),
                    "average": sum(rates) / len(rates),
                    "highest": max(rates),
                    "lowest": min(rates),
                    "latest": rates[0],
                    "first_data_time": str(data[-1].get('time')),
                    "latest_data_time": str(data[0].get('time'))
                }
            }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/compare")
def compare_rates(request: CompareRequest):
    """
    Compare funding rates across multiple symbols
    
    Request body:
    ```json
    {
        "symbols": ["btcusdt", "ethusdt"],
        "days": 3
    }
    ```
    """
    try:
        results = []
        
        for symbol in request.symbols:
            try:
                clean_symbol = symbol.lower().replace('-', '').replace('_', '')
                table_name = f"{clean_symbol}_1d"
                
                with get_db_connection() as conn, conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
                    cur.execute("""
                        SELECT table_name FROM information_schema.tables 
                        WHERE table_schema = 'funding_rate' AND table_name = %s
                    """, (table_name,))
                    
                    if not cur.fetchone():
                        results.append({
                            'symbol': symbol,
                            'error': 'No data available'
                        })
                        continue
                    
                    cur.execute(f"""
                        SELECT * FROM funding_rate."{table_name}"
                        WHERE time >= NOW() - INTERVAL '{request.days} days'
                        ORDER BY time DESC
                        LIMIT 100
                    """)
                    
                    data = list(cur.fetchall())
                    
                    if data:
                        latest = data[0]
                        results.append({
                            'symbol': symbol,
                            'latest_funding_rate': float(latest.get('funding_rate', 0)),
                            'time': str(latest.get('time')),
                            'data_points': len(data)
                        })
                    else:
                        results.append({
                            'symbol': symbol,
                            'error': 'No data available'
                        })
            except Exception as e:
                results.append({
                    'symbol': symbol,
                    'error': str(e)
                })
        
        return {
            "success": True,
            "days": request.days,
            "count": len(results),
            "comparison": results
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
def custom_query(request: QueryRequest):
    """
    Execute custom SQL query (SELECT only)
    
    Request body:
    ```json
    {
        "query": "SELECT * FROM funding_rate.btcusdt_1d LIMIT 5"
    }
    ```
    """
    try:
        query_clean = request.query.strip().lower()
        
        # Security checks
        if not query_clean.startswith('select'):
            raise HTTPException(
                status_code=400,
                detail="Only SELECT queries are allowed"
            )
        
        dangerous_keywords = ['insert', 'update', 'delete', 'drop', 'alter', 'create', 'truncate']
        if any(keyword in query_clean for keyword in dangerous_keywords):
            raise HTTPException(
                status_code=400,
                detail="Query contains forbidden keywords"
            )
        
        with get_db_connection() as conn, conn.cursor(row_factory=psycopg.rows.dict_row) as cur:
            cur.execute(request.query)
            results = list(cur.fetchall())
            
            return {
                "success": True,
                "count": len(results),
                "data": results
            }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")

# ========================================
# RUN SERVER
# ========================================

if __name__ == "__main__":
    print("=" * 50)
    print(" CoinGlass Funding Rates API")
    print("=" * 50)
    print(" Server: http://0.0.0.0:8781")
    print(" Docs: http://0.0.0.0:8781/docs")
    print("=" * 50)
    import os
    port = int(os.environ.get("PORT", 8781))
    uvicorn.run(app, host="0.0.0.0", port=port)