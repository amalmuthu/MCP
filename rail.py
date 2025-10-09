#!/usr/bin/env python3
"""
Binance Trading Analytics MCP Server
Professional-grade Model Context Protocol server for Binance futures trading analytics.

This server provides two main tools:
1. funding_arbitrage - Find high funding rate opportunities for traders
2. liquidation_zones - Monitor liquidation clusters for risk management

Built with FastMCP framework for enterprise-grade reliability.
"""

import os
import sys
import logging
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastmcp import FastMCP
import psycopg
from psycopg.rows import dict_row

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stderr)]
)

logger = logging.getLogger(__name__)

# Database connection
PG_DSN = "postgresql://postgres:jX8%23L7p%40qT9v%21sF2wY4z%24KbM@172.232.113.97:5432/coinglass"

def get_db_connection():
    """Get database connection"""
    return psycopg.connect(PG_DSN, autocommit=True)

# Initialize FastMCP server
mcp = FastMCP(
    name="Binance Trading Analytics",
    version="1.0.0"
)

# =============================================================================
# CORE ANALYTICS FUNCTIONS
# =============================================================================

def analyze_funding_arbitrage(
    min_rate: float,
    timeframe: str,
    lookback_hours: int
) -> Dict[str, Any]:
    """
    Core function to analyze funding rate arbitrage opportunities.
    
    Args:
        min_rate: Minimum funding rate threshold (e.g., 0.01 for 1%)
        timeframe: Time interval ('1h', '4h', '1d')
        lookback_hours: Hours to look back for analysis
    
    Returns:
        Dictionary with opportunities and analysis
    """
    with get_db_connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        # Get all funding rate tables for the timeframe
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
                    # Calculate average rate over period
                    avg_rate = sum(float(r['funding_rate'] or 0) for r in rows) / len(rows)
                    
                    opportunities.append({
                        'symbol': symbol.upper(),
                        'current_funding_rate': round(latest_rate * 100, 4),  # Convert to %
                        'average_funding_rate': round(avg_rate * 100, 4),
                        'current_price': float(latest['price'] or 0),
                        'last_update': str(latest['time']),
                        'annualized_rate': round(latest_rate * 365 * 3 * 100, 2),  # 3x per day
                        'signal': 'SHORT' if latest_rate > 0.01 else 'MONITOR'
                    })
                    
            except Exception as e:
                logger.debug(f"Error processing {table}: {e}")
                continue
        
        # Sort by funding rate
        opportunities.sort(key=lambda x: x['current_funding_rate'], reverse=True)
        
        return {
            'total_opportunities': len(opportunities),
            'min_rate_threshold': min_rate * 100,
            'timeframe': timeframe,
            'opportunities': opportunities[:20],  # Top 20
            'analysis_time': datetime.now().isoformat()
        }


def analyze_liquidation_zones(
    symbol: str,
    timeframe: str,
    periods: int
) -> Dict[str, Any]:
    """
    Core function to analyze liquidation zones and risk areas.
    
    Args:
        symbol: Trading pair (e.g., 'btcusdt')
        timeframe: Time interval ('1h', '4h', '1d')
        periods: Number of periods to analyze
    
    Returns:
        Dictionary with liquidation analysis
    """
    clean_symbol = symbol.lower().replace('-', '').replace('_', '')
    table_name = f"{clean_symbol}_{timeframe}"
    
    with get_db_connection() as conn, conn.cursor(row_factory=dict_row) as cur:
        # Check if table exists
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
        
        # Get liquidation data
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
        
        # Calculate statistics
        total_long_liq = sum(float(r['long_liquidation'] or 0) for r in rows)
        total_short_liq = sum(float(r['short_liquidation'] or 0) for r in rows)
        total_liq = total_long_liq + total_short_liq
        
        # Find high liquidation periods
        high_liq_periods = sorted(
            rows,
            key=lambda x: float(x['total_liquidation'] or 0),
            reverse=True
        )[:5]
        
        # Determine market sentiment
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
# MCP TOOLS - User-facing interface
# =============================================================================

@mcp.tool
def funding_arbitrage(
    min_rate: float = 0.01,
    timeframe: str = "1d",
    lookback_hours: int = 24
) -> str:
    """
    Find high funding rate arbitrage opportunities for profitable short positions.
    
    This tool helps traders identify cryptocurrencies with high funding rates where
    they can profit by shorting the perpetual contract and holding spot, earning
    the funding rate difference.
    
    Args:
        min_rate: Minimum funding rate threshold (default: 0.01 = 1%)
        timeframe: Time interval - '1h', '4h', '1d', '1w' (default: '1d')
        lookback_hours: Hours to analyze for trends (default: 24)
    
    Returns:
        Formatted analysis with top funding rate opportunities and trading signals
    
    Example:
        "Find funding opportunities above 2%"
        funding_arbitrage(min_rate=0.02, timeframe="4h")
    """
    try:
        result = analyze_funding_arbitrage(min_rate, timeframe, lookback_hours)
        
        if result['total_opportunities'] == 0:
            return f"""
No funding arbitrage opportunities found above {min_rate*100}% threshold.

Try lowering the min_rate or check a different timeframe.
Current settings: min_rate={min_rate}, timeframe={timeframe}
"""
        
        # Format output
        output = f"""
=== FUNDING ARBITRAGE OPPORTUNITIES ===
Threshold: >{min_rate*100}% | Timeframe: {timeframe} | Found: {result['total_opportunities']}

TOP OPPORTUNITIES:
"""
        for i, opp in enumerate(result['opportunities'][:10], 1):
            output += f"""
{i}. {opp['symbol']}
   Current Rate: {opp['current_funding_rate']}% (Annualized: {opp['annualized_rate']}%)
   Average Rate: {opp['average_funding_rate']}%
   Price: ${opp['current_price']:,.2f}
   Signal: {opp['signal']}
   Last Update: {opp['last_update']}
"""
        
        output += f"""
=== TRADING STRATEGY ===
1. Short perpetual futures on high funding rate pairs
2. Buy equivalent spot position (delta-neutral)
3. Collect funding rate payments every 8 hours
4. Monitor rates - close if rate drops significantly

Analysis Time: {result['analysis_time']}
"""
        return output
        
    except Exception as e:
        logger.error(f"Error in funding_arbitrage: {e}")
        return f"Error analyzing funding arbitrage: {str(e)}"


@mcp.tool
def liquidation_zones(
    symbol: str,
    timeframe: str = "1d",
    periods: int = 30
) -> str:
    """
    Monitor liquidation clusters and high-risk zones for risk management.
    
    This tool helps risk managers identify dangerous price levels with high
    liquidation activity, assess market sentiment, and manage position risk.
    
    Args:
        symbol: Trading pair (e.g., 'BTCUSDT', 'ETHUSDT')
        timeframe: Time interval - '1h', '4h', '1d' (default: '1d')
        periods: Number of periods to analyze (default: 30)
    
    Returns:
        Formatted analysis with liquidation zones, sentiment, and risk assessment
    
    Example:
        "Show liquidation zones for Bitcoin"
        liquidation_zones(symbol="BTCUSDT", timeframe="4h", periods=48)
    """
    try:
        result = analyze_liquidation_zones(symbol, timeframe, periods)
        
        if 'error' in result:
            return f"""
ERROR: {result['error']}

Symbol: {result['symbol']}
Available symbols: Use list_symbols() to see available pairs
"""
        
        # Format output
        output = f"""
=== LIQUIDATION ZONE ANALYSIS ===
Symbol: {result['symbol']} | Timeframe: {timeframe} | Periods: {periods}

=== OVERALL STATISTICS ===
Total Liquidations: ${result['total_liquidations']:,.2f}
Long Liquidations: ${result['long_liquidations']:,.2f} ({result['long_percentage']}%)
Short Liquidations: ${result['short_liquidations']:,.2f} ({result['short_percentage']}%)

Market Sentiment: {result['market_sentiment']}
Risk Level: {result['risk_level']}
Current Price: ${result['current_price']:,.2f}

=== HIGH LIQUIDATION ZONES (Top 5) ===
"""
        for i, zone in enumerate(result['high_liquidation_zones'], 1):
            output += f"""
{i}. Time: {zone['time']}
   Price: ${zone['price']:,.2f}
   Total Liquidation: ${zone['total_liquidation']:,.2f}
   Long: ${zone['long_liquidation']:,.2f} | Short: ${zone['short_liquidation']:,.2f}
"""
        
        output += f"""
=== RISK MANAGEMENT RECOMMENDATIONS ===
"""
        if result['long_percentage'] > 60:
            output += """
- High long liquidations suggest downward pressure
- Consider tightening long position stops
- Watch for potential support at liquidation zones
- Short bias may be favorable
"""
        elif result['short_percentage'] > 60:
            output += """
- High short liquidations suggest upward pressure
- Consider tightening short position stops
- Watch for potential resistance at liquidation zones
- Long bias may be favorable
"""
        else:
            output += """
- Balanced liquidations suggest neutral market
- Standard risk management applies
- Monitor for changes in liquidation balance
- No strong directional bias
"""
        
        output += f"""
Analysis Time: {result['analysis_time']}
"""
        return output
        
    except Exception as e:
        logger.error(f"Error in liquidation_zones: {e}")
        return f"Error analyzing liquidation zones: {str(e)}"


# =============================================================================
# HELPER TOOLS
# =============================================================================

@mcp.tool
def list_symbols(limit: int = 50) -> str:
    """
    List available trading pairs with data.
    
    Args:
        limit: Maximum number of symbols to return (default: 50)
    
    Returns:
        List of available trading pairs
    """
    try:
        with get_db_connection() as conn, conn.cursor() as cur:
            cur.execute("""
                SELECT DISTINCT REGEXP_REPLACE(table_name, '_[0-9a-z]+$', '') as symbol
                FROM information_schema.tables 
                WHERE table_schema = 'binance_futures_funding_rate_history'
                ORDER BY symbol
                LIMIT %s
            """, (limit,))
            
            symbols = [row[0].upper() for row in cur.fetchall()]
            
            return f"""
Available Trading Pairs ({len(symbols)}):

{', '.join(symbols)}

Use these symbols with funding_arbitrage() or liquidation_zones()
"""
    except Exception as e:
        return f"Error listing symbols: {str(e)}"


@mcp.tool
def search_symbol(query: str) -> str:
    """
    Search for trading pairs by name.
    
    Args:
        query: Search term (e.g., 'BTC', 'ETH', 'SOL')
    
    Returns:
        Matching trading pairs
    """
    try:
        with get_db_connection() as conn, conn.cursor() as cur:
            cur.execute("""
                SELECT DISTINCT REGEXP_REPLACE(table_name, '_[0-9a-z]+$', '') as symbol
                FROM information_schema.tables 
                WHERE table_schema = 'binance_futures_funding_rate_history'
                AND table_name LIKE %s
                ORDER BY symbol
                LIMIT 20
            """, (f'%{query.lower()}%',))
            
            symbols = [row[0].upper() for row in cur.fetchall()]
            
            if not symbols:
                return f"No symbols found matching '{query}'"
            
            return f"""
Found {len(symbols)} symbols matching '{query}':

{', '.join(symbols)}
"""
    except Exception as e:
        return f"Error searching symbols: {str(e)}"


# =============================================================================
# MCP RESOURCES
# =============================================================================

@mcp.resource("binance://funding-data")
def funding_data_resource() -> str:
    """
    Funding rate dataset information for LLM discovery.
    """
    import json
    return json.dumps({
        "dataset": "binance_futures_funding_rate_history",
        "description": "Historical funding rates for Binance futures",
        "available_timeframes": ["1h", "4h", "6h", "12h", "1d", "1w"],
        "total_symbols": 398,
        "metrics": ["funding_rate", "price", "time"],
        "usage": "Use funding_arbitrage() to find opportunities"
    }, indent=2)


@mcp.resource("binance://liquidation-data")
def liquidation_data_resource() -> str:
    """
    Liquidation dataset information for LLM discovery.
    """
    import json
    return json.dumps({
        "dataset": "binance_futures_liquidation_aggregated_history",
        "description": "Aggregated liquidation data for risk analysis",
        "available_timeframes": ["1h", "4h", "1d"],
        "total_symbols": 60,
        "metrics": ["long_liquidation", "short_liquidation", "total_liquidation", "price"],
        "usage": "Use liquidation_zones() to monitor risk"
    }, indent=2)


# =============================================================================
# SERVER STARTUP
# =============================================================================

def main():
    """Main function to start the MCP server."""
    try:
        logger.info(" Starting Binance Trading Analytics MCP Server...")
        logger.info(" Available tools:")
        logger.info("   - funding_arbitrage: Find high funding rate opportunities")
        logger.info("   - liquidation_zones: Monitor liquidation risk areas")
        logger.info("   - list_symbols: Show available trading pairs")
        logger.info("   - search_symbol: Search for specific pairs")
        
        # Test database connection
        with get_db_connection() as conn, conn.cursor() as cur:
            cur.execute("SELECT 1")
            logger.info(" Database connection successful")
        
        # Run server
        logger.info(" Starting in stdio mode")
        logger.info(" Users can ask:")
        logger.info("   - 'Find funding arbitrage opportunities above 2%'")
        logger.info("   - 'Show liquidation zones for Bitcoin'")
        logger.info("   - 'What symbols are available?'")
        
        port = int(os.environ.get("PORT", 10000))

# Run server in HTTP mode for Railway
        logger.info(f"Starting HTTP server on port {port}")
        mcp.run(transport="sse", host="0.0.0.0", port=port)
            
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
    except Exception as e:
        logger.error(f" Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()