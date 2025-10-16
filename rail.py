#!/usr/bin/env python3
"""
Coinglass Trading Analytics REST API
Provides trading analytics via simple REST endpoints for MCP Worker
"""

import os
import sys
import logging
from typing import Optional, Dict, Any, List
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
# HELPER FUNCTIONS
# =============================================================================

def get_rate_interpretation(rate: float) -> str:
    """Interpret what the funding rate means"""
    rate_pct = rate * 100
    
    if rate_pct > 0.1:
        return f"High positive rate ({rate_pct:.3f}%) - Longs paying shorts. Market is bullish/overheated. Arbitrage opportunity."
    elif rate_pct > 0:
        return f"Positive rate ({rate_pct:.3f}%) - Longs paying shorts. Slight bullish sentiment."
    elif rate_pct < -0.1:
        return f"High negative rate ({rate_pct:.3f}%) - Shorts paying longs. Market is bearish/oversold."
    elif rate_pct < 0:
        return f"Negative rate ({rate_pct:.3f}%) - Shorts paying longs. Slight bearish sentiment."
    else:
        return "Neutral rate - Balanced market."


def get_scan_summary(results: List[Dict]) -> str:
    """Generate summary of scan results"""
    if not results:
        return "No opportunities found."
    
    positive = sum(1 for r in results if r['funding_rate'] > 0)
    high_rates = sum(1 for r in results if r['funding_rate'] > 0.1)
    
    summary = f"Found {len(results)} coins. {positive} with positive rates. "
    if high_rates > 0:
        summary += f"{high_rates} coins have rates above 0.1% (arbitrage opportunities)."
    
    return summary


def get_top_analysis(top_coins: List[Dict]) -> str:
    """Analyze top coins"""
    if not top_coins:
        return "No data available."
    
    highest = top_coins[0]
    return f"Highest rate: {highest['symbol']} at {highest['funding_rate']}% ({highest['annualized_rate']}% annualized). " \
           f"Top coins show {'strong bullish' if highest['funding_rate'] > 0.5 else 'moderate'} sentiment."


def get_comparison_analysis(comparison: List[Dict]) -> str:
    """Compare multiple symbols"""
    if len(comparison) < 2:
        return "Need at least 2 symbols to compare."
    
    sorted_comp = sorted(comparison, key=lambda x: x['current_funding_rate'], reverse=True)
    highest = sorted_comp[0]
    lowest = sorted_comp[-1]
    
    return f"{highest['symbol']} has the highest rate at {highest['current_funding_rate']}%. " \
           f"{lowest['symbol']} has the lowest at {lowest['current_funding_rate']}%."


def calculate_volatility(values: List[float]) -> float:
    """Calculate standard deviation (volatility)"""
    if len(values) < 2:
        return 0
    
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return (variance ** 0.5) * 100  # As percentage


def get_liquidation_interpretation(long_pct: float, short_pct: float, total: float) -> Dict[str, str]:
    """Interpret liquidation data"""
    if long_pct > 70:
        sentiment = "STRONGLY BEARISH"
        interpretation = "Heavy long liquidations indicate strong downward pressure. Longs are getting rekt."
        risk_level = "HIGH"
    elif long_pct > 60:
        sentiment = "BEARISH"
        interpretation = "More longs getting liquidated. Bearish pressure dominant."
        risk_level = "ELEVATED"
    elif short_pct > 70:
        sentiment = "STRONGLY BULLISH"
        interpretation = "Heavy short liquidations indicate strong upward pressure. Shorts are getting squeezed."
        risk_level = "HIGH"
    elif short_pct > 60:
        sentiment = "BULLISH"
        interpretation = "More shorts getting liquidated. Bullish pressure dominant."
        risk_level = "ELEVATED"
    else:
        sentiment = "NEUTRAL"
        interpretation = "Balanced liquidations. No clear directional bias."
        risk_level = "MODERATE"
    
    return {
        "sentiment": sentiment,
        "interpretation": interpretation,
        "risk_level": risk_level
    }

# =============================================================================
# FUNDING RATE ANALYSIS 
# =============================================================================

def analyze_funding_rate(
    query_type: str,
    symbol: Optional[str] = None,
    timeframe: str = "1d",
    min_threshold: Optional[float] = None,
    lookback_hours: int = 168,
    limit: int = 20
) -> Dict[str, Any]:
    """
    Comprehensive funding rate analysis tool.
    
    Query Types:
    - 'current': Get current funding rate for specific symbol(s)
    - 'scan': Scan all coins for rates above/below threshold
    - 'trend': Show funding rate history/trend
    - 'top': Get top N coins by funding rate
    - 'compare': Compare multiple symbols
    - 'stats': Statistical analysis (avg, max, min, volatility)
    """
    
    with get_db_connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        
        # =====================================================================
        # QUERY TYPE 1: CURRENT RATE FOR SPECIFIC SYMBOL
        # =====================================================================
        if query_type == "current" and symbol:
            clean_symbol = symbol.lower().replace('-', '').replace('_', '')
            table_name = f"{clean_symbol}_{timeframe}"
            
            try:
                cur.execute(f"""
                    SELECT 
                        fr.close as funding_rate,
                        fr.time,
                        COALESCE(pr.close, '0') as price
                    FROM binance_futures_funding_rate_history."{table_name}" fr
                    LEFT JOIN binance_futures_price_history."{table_name}" pr
                        ON fr.time = pr.time
                    ORDER BY fr.time DESC
                    LIMIT 1
                """)
                
                row = cur.fetchone()
                if not row:
                    return {"error": f"No data for {symbol}"}
                
                rate = float(row['funding_rate'] or 0)
                price = float(row['price'] or 0)
                
                return {
                    "query_type": "current",
                    "symbol": symbol.upper(),
                    "timeframe": timeframe,
                    "current_funding_rate": round(rate * 100, 6),
                    "current_price": round(price, 2),
                    "annualized_rate": round(rate * 365 * 3 * 100, 2),
                    "last_update": str(row['time']),
                    "interpretation": get_rate_interpretation(rate)
                }
                
            except Exception as e:
                return {"error": f"Error fetching data for {symbol}: {str(e)}"}
        
        # =====================================================================
        # QUERY TYPE 2: SCAN ALL COINS (with optional threshold)
        # =====================================================================
        elif query_type == "scan":
            cur.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'binance_futures_funding_rate_history'
                AND table_name LIKE %s
                LIMIT 200
            """, (f'%_{timeframe}',))
            
            tables = [row['table_name'] for row in cur.fetchall()]
            results = []
            
            for table in tables:
                try:
                    symbol_name = table.replace(f'_{timeframe}', '')
                    
                    cur.execute(f"""
                        SELECT 
                            fr.close as funding_rate,
                            fr.time,
                            COALESCE(pr.close, '0') as price
                        FROM binance_futures_funding_rate_history."{table}" fr
                        LEFT JOIN binance_futures_price_history."{table}" pr
                            ON fr.time = pr.time
                        ORDER BY fr.time DESC
                        LIMIT 1
                    """)
                    
                    row = cur.fetchone()
                    if not row:
                        continue
                    
                    rate = float(row['funding_rate'] or 0)
                    price = float(row['price'] or 0)
                    
                    # Apply threshold filter if provided
                    if min_threshold is not None and rate < min_threshold:
                        continue
                    
                    results.append({
                        'symbol': symbol_name.upper(),
                        'funding_rate': round(rate * 100, 6),
                        'price': round(price, 2),
                        'annualized_rate': round(rate * 365 * 3 * 100, 2),
                        'last_update': str(row['time'])
                    })
                    
                except Exception as e:
                    logger.debug(f"Error processing {table}: {e}")
                    continue
            
            # Sort by funding rate
            results.sort(key=lambda x: x['funding_rate'], reverse=True)
            
            return {
                "query_type": "scan",
                "timeframe": timeframe,
                "threshold": min_threshold * 100 if min_threshold else None,
                "total_found": len(results),
                "results": results[:limit],
                "summary": get_scan_summary(results)
            }
        
        # =====================================================================
        # QUERY TYPE 3: HISTORICAL TREND
        # =====================================================================
        elif query_type == "trend" and symbol:
            clean_symbol = symbol.lower().replace('-', '').replace('_', '')
            table_name = f"{clean_symbol}_{timeframe}"
            
            try:
                cur.execute(f"""
                SELECT 
                    fr.close as funding_rate,
                    fr.time,
                    COALESCE(pr.close, '0') as price
                FROM binance_futures_funding_rate_history."{table_name}" fr
                LEFT JOIN binance_futures_price_history."{table_name}" pr
                    ON fr.time = pr.time
                WHERE fr.time::timestamp >= NOW() - INTERVAL '{lookback_hours} hours'
                ORDER BY fr.time DESC
                LIMIT 50
            """)

                
                rows = cur.fetchall()
                if not rows:
                    return {"error": f"No historical data for {symbol}"}
                
                history = []
                rates = []
                for row in rows:
                    rate = float(row['funding_rate'] or 0)
                    rates.append(rate)
                    history.append({
                        'time': str(row['time']),
                        'funding_rate': round(rate * 100, 6),
                        'price': round(float(row['price'] or 0), 2)
                    })
                
                # Calculate trend
                trend_direction = "increasing" if rates[0] > rates[-1] else "decreasing"
                avg_rate = sum(rates) / len(rates)
                
                return {
                    "query_type": "trend",
                    "symbol": symbol.upper(),
                    "timeframe": timeframe,
                    "lookback_hours": lookback_hours,
                    "current_rate": round(rates[0] * 100, 6),
                    "average_rate": round(avg_rate * 100, 6),
                    "trend": trend_direction,
                    "data_points": len(history),
                    "history": history[:20],
                    "statistics": {
                        "max": round(max(rates) * 100, 6),
                        "min": round(min(rates) * 100, 6),
                        "volatility": round(calculate_volatility(rates), 6)
                    }
                }
                
            except Exception as e:
                return {"error": f"Error fetching trend for {symbol}: {str(e)}"}
        
        # =====================================================================
        # QUERY TYPE 4: TOP N COINS
        # =====================================================================
        elif query_type == "top":
            scan_result = analyze_funding_rate("scan", timeframe=timeframe, limit=limit)
            
            if "error" in scan_result:
                return scan_result
            
            return {
                "query_type": "top",
                "timeframe": timeframe,
                "limit": limit,
                "top_coins": scan_result["results"][:limit],
                "analysis": get_top_analysis(scan_result["results"][:limit])
            }
        
        # =====================================================================
        # QUERY TYPE 5: COMPARE MULTIPLE SYMBOLS
        # =====================================================================
        elif query_type == "compare" and symbol:
            symbols = [s.strip() for s in symbol.split(',')]
            comparison = []
            
            for sym in symbols:
                result = analyze_funding_rate("current", symbol=sym, timeframe=timeframe)
                if "error" not in result:
                    comparison.append(result)
            
            if not comparison:
                return {"error": "No data found for provided symbols"}
            
            return {
                "query_type": "compare",
                "timeframe": timeframe,
                "symbols": [c['symbol'] for c in comparison],
                "comparison": comparison,
                "analysis": get_comparison_analysis(comparison)
            }
        
        # =====================================================================
        # QUERY TYPE 6: STATISTICS
        # =====================================================================
        elif query_type == "stats":
            cur.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'binance_futures_funding_rate_history'
                AND table_name LIKE %s
                LIMIT 100
            """, (f'%_{timeframe}',))
            
            tables = [row['table_name'] for row in cur.fetchall()]
            all_rates = []
            positive_count = 0
            negative_count = 0
            
            for table in tables:
                try:
                    cur.execute(f"""
                        SELECT close as funding_rate
                        FROM binance_futures_funding_rate_history."{table}"
                        ORDER BY time DESC
                        LIMIT 1
                    """)
                    
                    row = cur.fetchone()
                    if row and row['funding_rate']:
                        rate = float(row['funding_rate'])
                        all_rates.append(rate)
                        if rate > 0:
                            positive_count += 1
                        else:
                            negative_count += 1
                            
                except Exception as e:
                    continue
            
            if not all_rates:
                return {"error": "No data available for statistics"}
            
            avg_rate = sum(all_rates) / len(all_rates)
            
            return {
                "query_type": "stats",
                "timeframe": timeframe,
                "total_coins": len(all_rates),
                "positive_funding": positive_count,
                "negative_funding": negative_count,
                "positive_percentage": round(positive_count / len(all_rates) * 100, 2),
                "average_rate": round(avg_rate * 100, 6),
                "max_rate": round(max(all_rates) * 100, 6),
                "min_rate": round(min(all_rates) * 100, 6),
                "volatility": round(calculate_volatility(all_rates), 6),
                "market_sentiment": "Bullish" if positive_count > negative_count else "Bearish"
            }
        
        else:
            return {
                "error": "Invalid query type or missing parameters",
                "valid_types": ["current", "scan", "trend", "top", "compare", "stats"]
            }


# =============================================================================
# LIQUIDATION ANALYSIS (NEW COMPREHENSIVE TOOL)
# =============================================================================
def analyze_liquidation(
    query_type: str,
    symbol: Optional[str] = None,
    timeframe: str = "1d",
    min_threshold: Optional[float] = None,
    lookback_hours: int = 168,
    limit: int = 20,
    use_individual: bool = False
) -> Dict[str, Any]:
    """
    Comprehensive liquidation analysis tool.
    
    Query Types:
    - 'current': Current liquidation activity for symbol
    - 'zones': Liquidation clusters at price levels
    - 'trend': Historical liquidation patterns
    - 'compare': Compare liquidations across symbols
    - 'stats': Market-wide liquidation statistics
    - 'whale': Large individual liquidations
    - 'scan': Find coins with high liquidation activity
    - 'cascade': Detect liquidation cascade events
    """
    
    # ============================================================================
    # HELPER FUNCTIONS FOR TABLE NAMING
    # ============================================================================
    
    def get_liq_table_name(sym: str, tf: str, individual: bool) -> str:
        """Get correct liquidation table name based on schema"""
        clean = sym.lower().replace('-', '').replace('_', '')
        if not individual:
            # Aggregated schema: btc_1d (remove 'usdt')
            clean = clean.replace('usdt', '')
        # Individual schema: btcusdt_1d (keep 'usdt')
        return f"{clean}_{tf}"
    
    def get_price_table_name(sym: str, tf: str) -> str:
        """Get price table name - always includes 'usdt'"""
        clean = sym.lower().replace('-', '').replace('_', '')
        return f"{clean}_{tf}"
    
    # ============================================================================
    # SCHEMA AND COLUMN SELECTION
    # ============================================================================
    
    schema = 'binance_futures_liquidation_history' if use_individual else 'binance_futures_liquidation_aggregated_history'
    long_col = 'long_liquidation_usd' if use_individual else 'aggregated_long_liquidation_usd'
    short_col = 'short_liquidation_usd' if use_individual else 'aggregated_short_liquidation_usd'
    
    with get_db_connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        
        # ====================================================================
        # QUERY TYPE 1: CURRENT LIQUIDATION ACTIVITY
        # ====================================================================
        if query_type == "current" and symbol:
            liq_table = get_liq_table_name(symbol, timeframe, use_individual)
            price_table = get_price_table_name(symbol, timeframe)
            
            try:
                cur.execute(f"""
                    SELECT 
                        liq.{long_col} as long_liquidation,
                        liq.{short_col} as short_liquidation,
                        liq.time,
                        COALESCE(pr.close, '0') as price
                    FROM {schema}."{liq_table}" liq
                    LEFT JOIN binance_futures_price_history."{price_table}" pr
                        ON liq.time = pr.time
                    ORDER BY liq.time DESC
                    LIMIT 10
                """)
                
                rows = cur.fetchall()
                if not rows:
                    return {"error": f"No liquidation data for {symbol}"}
                
                latest = rows[0]
                long_liq = float(latest['long_liquidation'] or 0)
                short_liq = float(latest['short_liquidation'] or 0)
                total_liq = long_liq + short_liq
                
                long_pct = (long_liq / total_liq * 100) if total_liq > 0 else 0
                short_pct = (short_liq / total_liq * 100) if total_liq > 0 else 0
                
                # Get recent activity
                recent_activity = []
                for row in rows[:5]:
                    l_liq = float(row['long_liquidation'] or 0)
                    s_liq = float(row['short_liquidation'] or 0)
                    recent_activity.append({
                        'time': str(row['time']),
                        'long_liquidation': round(l_liq, 2),
                        'short_liquidation': round(s_liq, 2),
                        'total': round(l_liq + s_liq, 2),
                        'price': round(float(row['price'] or 0), 2)
                    })
                
                analysis = get_liquidation_interpretation(long_pct, short_pct, total_liq)
                
                return {
                    "query_type": "current",
                    "symbol": symbol.upper(),
                    "timeframe": timeframe,
                    "current_liquidations": {
                        "long": round(long_liq, 2),
                        "short": round(short_liq, 2),
                        "total": round(total_liq, 2),
                        "long_percentage": round(long_pct, 2),
                        "short_percentage": round(short_pct, 2)
                    },
                    "current_price": round(float(latest['price'] or 0), 2),
                    "last_update": str(latest['time']),
                    "recent_activity": recent_activity,
                    "market_sentiment": analysis["sentiment"],
                    "interpretation": analysis["interpretation"],
                    "risk_level": analysis["risk_level"]
                }
                
            except Exception as e:
                return {"error": f"Error fetching liquidation data for {symbol}: {str(e)}"}
        
        # ====================================================================
        # QUERY TYPE 2: LIQUIDATION ZONES (with price levels)
        # ====================================================================
        elif query_type == "zones" and symbol:
            liq_table = get_liq_table_name(symbol, timeframe, use_individual)
            price_table = get_price_table_name(symbol, timeframe)
            
            try:
                cur.execute(f"""
                    SELECT 
                        liq.{long_col} as long_liquidation,
                        liq.{short_col} as short_liquidation,
                        liq.time,
                        COALESCE(pr.close, '0') as price
                    FROM {schema}."{liq_table}" liq
                    LEFT JOIN binance_futures_price_history."{price_table}" pr
                        ON liq.time = pr.time
                    WHERE liq.time::timestamp >= NOW() - INTERVAL '{lookback_hours} hours'
                    ORDER BY liq.time DESC
                    LIMIT 100
                """)
                
                rows = cur.fetchall()
                if not rows:
                    return {"error": f"No liquidation zone data for {symbol}"}
                
                # Calculate totals
                total_long = sum(float(r['long_liquidation'] or 0) for r in rows)
                total_short = sum(float(r['short_liquidation'] or 0) for r in rows)
                total = total_long + total_short
                
                long_pct = (total_long / total * 100) if total > 0 else 0
                short_pct = (total_short / total * 100) if total > 0 else 0
                
                # Find high liquidation zones
                zones_data = []
                for row in rows:
                    l_liq = float(row['long_liquidation'] or 0)
                    s_liq = float(row['short_liquidation'] or 0)
                    total_liq = l_liq + s_liq
                    
                    if total_liq > 0:
                        zones_data.append({
                            'time': str(row['time']),
                            'price': round(float(row['price'] or 0), 2),
                            'long_liquidation': round(l_liq, 2),
                            'short_liquidation': round(s_liq, 2),
                            'total_liquidation': round(total_liq, 2)
                        })
                
                # Sort by total liquidation and get top zones
                high_zones = sorted(zones_data, key=lambda x: x['total_liquidation'], reverse=True)[:10]
                
                analysis = get_liquidation_interpretation(long_pct, short_pct, total)
                
                return {
                    "query_type": "zones",
                    "symbol": symbol.upper(),
                    "timeframe": timeframe,
                    "lookback_hours": lookback_hours,
                    "summary": {
                        "total_liquidations": round(total, 2),
                        "long_liquidations": round(total_long, 2),
                        "short_liquidations": round(total_short, 2),
                        "long_percentage": round(long_pct, 2),
                        "short_percentage": round(short_pct, 2)
                    },
                    "current_price": high_zones[0]['price'] if high_zones else 0,
                    "high_liquidation_zones": high_zones,
                    "market_sentiment": analysis["sentiment"],
                    "interpretation": analysis["interpretation"],
                    "risk_level": analysis["risk_level"]
                }
                
            except Exception as e:
                return {"error": f"Error fetching liquidation zones for {symbol}: {str(e)}"}
        
        # ====================================================================
        # QUERY TYPE 3: HISTORICAL TREND
        # ====================================================================
        elif query_type == "trend" and symbol:
            liq_table = get_liq_table_name(symbol, timeframe, use_individual)
            
            try:
                cur.execute(f"""
                    SELECT 
                        {long_col} as long_liquidation,
                        {short_col} as short_liquidation,
                        time
                    FROM {schema}."{liq_table}"
                    WHERE time::timestamp >= NOW() - INTERVAL '{lookback_hours} hours'
                    ORDER BY time DESC
                    LIMIT 50
                """)
                
                rows = cur.fetchall()
                if not rows:
                    return {"error": f"No trend data for {symbol}"}
                
                history = []
                total_liquidations = []
                
                for row in rows:
                    l_liq = float(row['long_liquidation'] or 0)
                    s_liq = float(row['short_liquidation'] or 0)
                    total = l_liq + s_liq
                    
                    total_liquidations.append(total)
                    history.append({
                        'time': str(row['time']),
                        'long_liquidation': round(l_liq, 2),
                        'short_liquidation': round(s_liq, 2),
                        'total_liquidation': round(total, 2)
                    })
                
                # Calculate trend
                trend_direction = "increasing" if total_liquidations[0] > total_liquidations[-1] else "decreasing"
                avg_liquidation = sum(total_liquidations) / len(total_liquidations)
                
                return {
                    "query_type": "trend",
                    "symbol": symbol.upper(),
                    "timeframe": timeframe,
                    "lookback_hours": lookback_hours,
                    "current_liquidation": round(total_liquidations[0], 2),
                    "average_liquidation": round(avg_liquidation, 2),
                    "trend": trend_direction,
                    "data_points": len(history),
                    "history": history[:20],
                    "statistics": {
                        "max": round(max(total_liquidations), 2),
                        "min": round(min(total_liquidations), 2),
                        "volatility": round(calculate_volatility(total_liquidations), 2)
                    }
                }
                
            except Exception as e:
                return {"error": f"Error fetching trend for {symbol}: {str(e)}"}
        
        # ====================================================================
        # QUERY TYPE 4: COMPARE MULTIPLE SYMBOLS
        # ====================================================================
        elif query_type == "compare" and symbol:
            symbols = [s.strip() for s in symbol.split(',')]
            comparison = []
            
            for sym in symbols:
                result = analyze_liquidation("current", symbol=sym, timeframe=timeframe, use_individual=use_individual)
                if "error" not in result:
                    comparison.append({
                        'symbol': result['symbol'],
                        'total_liquidations': result['current_liquidations']['total'],
                        'long_percentage': result['current_liquidations']['long_percentage'],
                        'short_percentage': result['current_liquidations']['short_percentage'],
                        'sentiment': result['market_sentiment']
                    })
            
            if not comparison:
                return {"error": "No data found for provided symbols"}
            
            # Sort by total liquidations
            comparison.sort(key=lambda x: x['total_liquidations'], reverse=True)
            
            return {
                "query_type": "compare",
                "timeframe": timeframe,
                "symbols": [c['symbol'] for c in comparison],
                "comparison": comparison,
                "analysis": f"Highest liquidations: {comparison[0]['symbol']} (${comparison[0]['total_liquidations']:,.2f})"
            }
        
        # ====================================================================
        # QUERY TYPE 5: MARKET-WIDE STATISTICS
        # ====================================================================
        elif query_type == "stats":
            cur.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = %s
                AND table_name LIKE %s
                LIMIT 100
            """, (schema, f'%_{timeframe}'))
            
            tables = [row['table_name'] for row in cur.fetchall()]
            all_liquidations = []
            total_long = 0
            total_short = 0
            
            for table in tables:
                try:
                    cur.execute(f"""
                        SELECT {long_col} as long_liq, {short_col} as short_liq
                        FROM {schema}."{table}"
                        ORDER BY time DESC
                        LIMIT 1
                    """)
                    
                    row = cur.fetchone()
                    if row:
                        l_liq = float(row['long_liq'] or 0)
                        s_liq = float(row['short_liq'] or 0)
                        total = l_liq + s_liq
                        
                        if total > 0:
                            all_liquidations.append(total)
                            total_long += l_liq
                            total_short += s_liq
                            
                except Exception as e:
                    continue
            
            if not all_liquidations:
                return {"error": "No liquidation data available"}
            
            total = total_long + total_short
            long_pct = (total_long / total * 100) if total > 0 else 0
            short_pct = (total_short / total * 100) if total > 0 else 0
            
            analysis = get_liquidation_interpretation(long_pct, short_pct, total)
            
            return {
                "query_type": "stats",
                "timeframe": timeframe,
                "total_coins": len(all_liquidations),
                "total_liquidations": round(total, 2),
                "long_liquidations": round(total_long, 2),
                "short_liquidations": round(total_short, 2),
                "long_percentage": round(long_pct, 2),
                "short_percentage": round(short_pct, 2),
                "average_liquidation": round(sum(all_liquidations) / len(all_liquidations), 2),
                "max_liquidation": round(max(all_liquidations), 2),
                "min_liquidation": round(min(all_liquidations), 2),
                "market_sentiment": analysis["sentiment"],
                "interpretation": analysis["interpretation"],
                "risk_level": analysis["risk_level"]
            }
        
        # ====================================================================
        # QUERY TYPE 6: WHALE LIQUIDATIONS (large events)
        # ====================================================================
        elif query_type == "whale":
            # Use individual liquidation history for whale tracking
            cur.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'binance_futures_liquidation_history'
                AND table_name LIKE %s
                LIMIT 200
            """, (f'%_{timeframe}',))
            
            tables = [row['table_name'] for row in cur.fetchall()]
            whale_liquidations = []
            threshold = min_threshold if min_threshold else 100000  # Default $100k
            
            for table in tables:
                try:
                    # For individual schema, symbol name includes 'usdt'
                    symbol_name = table.replace(f'_{timeframe}', '')
                    
                    cur.execute(f"""
                        SELECT 
                            long_liquidation_usd as long_liq,
                            short_liquidation_usd as short_liq,
                            time
                        FROM binance_futures_liquidation_history."{table}"
                        WHERE time::timestamp >= NOW() - INTERVAL '{lookback_hours} hours'
                        ORDER BY time DESC
                        LIMIT 50
                    """)
                    
                    rows = cur.fetchall()
                    for row in rows:
                        l_liq = float(row['long_liq'] or 0)
                        s_liq = float(row['short_liq'] or 0)
                        
                        if l_liq >= threshold:
                            whale_liquidations.append({
                                'symbol': symbol_name.upper(),
                                'type': 'LONG',
                                'amount': round(l_liq, 2),
                                'time': str(row['time'])
                            })
                        
                        if s_liq >= threshold:
                            whale_liquidations.append({
                                'symbol': symbol_name.upper(),
                                'type': 'SHORT',
                                'amount': round(s_liq, 2),
                                'time': str(row['time'])
                            })
                            
                except Exception as e:
                    logger.debug(f"Error processing {table}: {e}")
                    continue
            
            # Sort by amount
            whale_liquidations.sort(key=lambda x: x['amount'], reverse=True)
            
            return {
                "query_type": "whale",
                "timeframe": timeframe,
                "threshold": threshold,
                "lookback_hours": lookback_hours,
                "total_whale_liquidations": len(whale_liquidations),
                "whale_liquidations": whale_liquidations[:limit],
                "largest": whale_liquidations[0] if whale_liquidations else None
            }
        
        # ====================================================================
        # QUERY TYPE 7: SCAN FOR HIGH LIQUIDATION ACTIVITY
        # ====================================================================
        elif query_type == "scan":
            cur.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = %s
                AND table_name LIKE %s
                LIMIT 200
            """, (schema, f'%_{timeframe}'))
            
            tables = [row['table_name'] for row in cur.fetchall()]
            results = []
            threshold = min_threshold if min_threshold else 0
            
            for table in tables:
                try:
                    # Extract symbol name based on schema
                    if use_individual:
                        symbol_name = table.replace(f'_{timeframe}', '')
                    else:
                        # For aggregated, add 'usdt' back for display
                        symbol_name = table.replace(f'_{timeframe}', '') + 'usdt'
                    
                    cur.execute(f"""
                        SELECT 
                            {long_col} as long_liq,
                            {short_col} as short_liq,
                            time
                        FROM {schema}."{table}"
                        ORDER BY time DESC
                        LIMIT 1
                    """)
                    
                    row = cur.fetchone()
                    if not row:
                        continue
                    
                    l_liq = float(row['long_liq'] or 0)
                    s_liq = float(row['short_liq'] or 0)
                    total = l_liq + s_liq
                    
                    # Apply threshold filter
                    if total < threshold:
                        continue
                    
                    long_pct = (l_liq / total * 100) if total > 0 else 0
                    short_pct = (s_liq / total * 100) if total > 0 else 0
                    
                    results.append({
                        'symbol': symbol_name.upper(),
                        'total_liquidation': round(total, 2),
                        'long_liquidation': round(l_liq, 2),
                        'short_liquidation': round(s_liq, 2),
                        'long_percentage': round(long_pct, 2),
                        'short_percentage': round(short_pct, 2),
                        'dominant_side': 'LONG' if long_pct > short_pct else 'SHORT',
                        'last_update': str(row['time'])
                    })
                    
                except Exception as e:
                    logger.debug(f"Error processing {table}: {e}")
                    continue
            
            # Sort by total liquidation
            results.sort(key=lambda x: x['total_liquidation'], reverse=True)
            
            return {
                "query_type": "scan",
                "timeframe": timeframe,
                "threshold": threshold,
                "total_found": len(results),
                "results": results[:limit],
                "summary": f"Found {len(results)} coins with liquidations. Top: {results[0]['symbol']} (${results[0]['total_liquidation']:,.2f})" if results else "No results"
            }
        
        # ====================================================================
        # QUERY TYPE 8: LIQUIDATION CASCADE DETECTION
        # ====================================================================
        elif query_type == "cascade" and symbol:
            liq_table = get_liq_table_name(symbol, timeframe, use_individual)
            price_table = get_price_table_name(symbol, timeframe)
            
            try:
                cur.execute(f"""
                    SELECT 
                        liq.{long_col} as long_liq,
                        liq.{short_col} as short_liq,
                        liq.time,
                        COALESCE(pr.close, '0') as price,
                        COALESCE(LAG(pr.close) OVER (ORDER BY liq.time), pr.close) as prev_price
                    FROM {schema}."{liq_table}" liq
                    LEFT JOIN binance_futures_price_history."{price_table}" pr
                        ON liq.time = pr.time
                    WHERE liq.time::timestamp >= NOW() - INTERVAL '{lookback_hours} hours'
                    ORDER BY liq.time DESC
                    LIMIT 100
                """)
                
                rows = cur.fetchall()
                if not rows:
                    return {"error": f"No cascade data for {symbol}"}
                
                cascades = []
                
                for i, row in enumerate(rows):
                    l_liq = float(row['long_liq'] or 0)
                    s_liq = float(row['short_liq'] or 0)
                    total_liq = l_liq + s_liq
                    
                    price = float(row['price'] or 0)
                    prev_price = float(row['prev_price'] or price)
                    price_change_pct = ((price - prev_price) / prev_price * 100) if prev_price > 0 else 0
                    
                    # Detect cascade: large liquidation + significant price move
                    if total_liq > 0 and abs(price_change_pct) > 2:  # 2% price move
                        cascade_type = "LONG_SQUEEZE" if l_liq > s_liq and price_change_pct < 0 else \
                                      "SHORT_SQUEEZE" if s_liq > l_liq and price_change_pct > 0 else \
                                      "MIXED"
                        
                        cascades.append({
                            'time': str(row['time']),
                            'type': cascade_type,
                            'total_liquidation': round(total_liq, 2),
                            'long_liquidation': round(l_liq, 2),
                            'short_liquidation': round(s_liq, 2),
                            'price': round(price, 2),
                            'price_change_pct': round(price_change_pct, 2)
                        })
                
                return {
                    "query_type": "cascade",
                    "symbol": symbol.upper(),
                    "timeframe": timeframe,
                    "lookback_hours": lookback_hours,
                    "cascade_events_detected": len(cascades),
                    "cascades": cascades[:10],
                    "interpretation": f"Detected {len(cascades)} potential cascade events with >2% price moves"
                }
                
            except Exception as e:
                return {"error": f"Error detecting cascades for {symbol}: {str(e)}"}
        
        else:
            return {
                "error": "Invalid query type or missing parameters",
                "valid_types": ["current", "zones", "trend", "compare", "stats", "whale", "scan", "cascade"]
            }
# =============================================================================
# REST API ENDPOINTS
# =============================================================================

class FundingRateRequest(BaseModel):
    query_type: str
    symbol: Optional[str] = None
    timeframe: str = "1d"
    min_threshold: Optional[float] = None
    lookback_hours: int = 168
    limit: int = 20

class LiquidationRequest(BaseModel):
    query_type: str
    symbol: Optional[str] = None
    timeframe: str = "1d"
    min_threshold: Optional[float] = None
    lookback_hours: int = 168
    limit: int = 20
    use_individual: bool = False

class ListSymbolsRequest(BaseModel):
    limit: int = 50

class SearchSymbolRequest(BaseModel):
    query: str


@app.post("/api/funding_rate")
async def api_funding_rate(req: FundingRateRequest):
    """
    Comprehensive funding rate analysis endpoint.
    
    Query types:
    - current: Get current rate for specific symbol
    - scan: Scan all coins (with optional threshold)
    - trend: Show historical trend
    - top: Get top N coins by rate
    - compare: Compare multiple symbols (comma-separated)
    - stats: Market-wide statistics
    """
    try:
        result = analyze_funding_rate(
            query_type=req.query_type,
            symbol=req.symbol,
            timeframe=req.timeframe,
            min_threshold=req.min_threshold,
            lookback_hours=req.lookback_hours,
            limit=req.limit
        )
        return result
    except Exception as e:
        logger.error(f"Error in funding_rate: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/liquidation")
async def api_liquidation(req: LiquidationRequest):
    """
    Comprehensive liquidation analysis endpoint.
    
    Query types:
    - current: Current liquidation activity for symbol
    - zones: Liquidation clusters at price levels
    - trend: Historical liquidation patterns
    - compare: Compare liquidations across symbols (comma-separated)
    - stats: Market-wide liquidation statistics
    - whale: Large individual liquidations (>threshold)
    - scan: Find coins with high liquidation activity
    - cascade: Detect liquidation cascade events
    """
    try:
        result = analyze_liquidation(
            query_type=req.query_type,
            symbol=req.symbol,
            timeframe=req.timeframe,
            min_threshold=req.min_threshold,
            lookback_hours=req.lookback_hours,
            limit=req.limit,
            use_individual=req.use_individual
        )
        return result
    except Exception as e:
        logger.error(f"Error in liquidation: {e}")
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
        "version": "2.0.0",
        "endpoints": {
            "funding_rate": "/api/funding_rate (comprehensive funding analysis)",
            "liquidation": "/api/liquidation (comprehensive liquidation analysis - NEW)",
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
        logger.info(" Starting Coinglass Trading Analytics REST API v2.0...")
        
        # Test database connection
        with get_db_connection() as conn, conn.cursor() as cur:
            cur.execute("SELECT 1")
            logger.info("Database connection successful")
        
        port = int(os.environ.get("PORT", 10000))
        
        logger.info(f" Starting REST API server on 0.0.0.0:{port}")
        logger.info(" API Documentation: /docs")
        logger.info(" New: Comprehensive funding_rate + liquidation tools")
        
        # Run FastAPI with uvicorn
        uvicorn.run(app, host="0.0.0.0", port=port)
            
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
    except Exception as e:
        logger.error(f" Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()