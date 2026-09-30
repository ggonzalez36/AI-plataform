package middleware

import (
	"context"
	"fmt"
	"log"
	"math"
	"net/http"
	"strconv"
	"sync"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/redis/go-redis/v9"
)

const tokenBucketLuaScript = `
local key = KEYS[1]
local capacity = tonumber(ARGV[1])
local fill_rate = tonumber(ARGV[2])
local now = tonumber(ARGV[3])
local requested = tonumber(ARGV[4])
local ttl = tonumber(ARGV[5])

local data = redis.call('HMGET', key, 'tokens', 'last_updated')
local tokens = tonumber(data[1])
local last_updated = tonumber(data[2])

if tokens == nil then
    tokens = capacity
    last_updated = now
else
    local delta = math.max(0, now - last_updated)
    local tokens_to_add = delta * fill_rate
    tokens = math.min(capacity, tokens + tokens_to_add)
    last_updated = now
end

if tokens >= requested then
    tokens = tokens - requested
    redis.call('HMSET', key, 'tokens', tokens, 'last_updated', last_updated)
    redis.call('EXPIRE', key, ttl)
    return {1, math.floor(tokens), 0}
else
    local missing = requested - tokens
    local retry_after = math.ceil(missing / fill_rate)
    if retry_after < 1 then
        retry_after = 1
    end
    redis.call('HMSET', key, 'tokens', tokens, 'last_updated', last_updated)
    redis.call('EXPIRE', key, ttl)
    return {0, math.floor(tokens), retry_after}
end
`

// MemoryBucket holds state for in-memory token bucket fallback
type MemoryBucket struct {
	tokens      float64
	lastUpdated time.Time
}

// RateLimiter manages distributed Redis and fallback in-memory rate limiting
type RateLimiter struct {
	redisClient   *redis.Client
	redisHealthy  bool
	luaScript     *redis.Script
	rateRPS       int
	capacityBurst int
	memMu         sync.Mutex
	memBuckets    map[string]*MemoryBucket
}

// NewRateLimiter initializes the distributed rate limiter with fallback resilience
func NewRateLimiter(redisClient *redis.Client, rateRPS, capacityBurst int) *RateLimiter {
	limiter := &RateLimiter{
		redisClient:   redisClient,
		redisHealthy:  false,
		luaScript:     redis.NewScript(tokenBucketLuaScript),
		rateRPS:       rateRPS,
		capacityBurst: capacityBurst,
		memBuckets:    make(map[string]*MemoryBucket),
	}

	if redisClient != nil {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		defer cancel()
		if err := redisClient.Ping(ctx).Err(); err == nil {
			limiter.redisHealthy = true
			log.Printf("🟢 Redis rate limiter initialized (%d req/s, burst: %d)", rateRPS, capacityBurst)
		} else {
			log.Printf("⚠️ Redis unavailable at startup (%v). Enabling resilient in-memory fallback.", err)
		}
	} else {
		log.Printf("ℹ️ No Redis client configured. Running in-memory rate limiter.")
	}

	return limiter
}

// IsRedisHealthy returns whether Redis connectivity is active
func (rl *RateLimiter) IsRedisHealthy() bool {
	if rl.redisClient == nil {
		return false
	}
	ctx, cancel := context.WithTimeout(context.Background(), 500*time.Millisecond)
	defer cancel()
	return rl.redisClient.Ping(ctx).Err() == nil
}

// Allow evaluates if a request with given key is allowed under the token bucket
func (rl *RateLimiter) Allow(ctx context.Context, key string) (allowed bool, remaining int, retryAfter int) {
	// Try Redis if available
	if rl.redisClient != nil {
		nowSec := float64(time.Now().UnixNano()) / 1e9
		ttl := int(math.Ceil(float64(rl.capacityBurst)/float64(rl.rateRPS))) + 10
		if ttl < 60 {
			ttl = 60
		}

		res, err := rl.luaScript.Run(ctx, rl.redisClient, []string{fmt.Sprintf("ratelimit:%s", key)},
			rl.capacityBurst, rl.rateRPS, nowSec, 1, ttl).Result()

		if err == nil {
			rl.redisHealthy = true
			if slice, ok := res.([]interface{}); ok && len(slice) >= 3 {
				allowedFlag := slice[0].(int64) == 1
				rem := int(slice[1].(int64))
				retry := int(slice[2].(int64))
				return allowedFlag, rem, retry
			}
		}
		// If Redis returned error, flag unhealthy and fallback to in-memory
		rl.redisHealthy = false
	}

	// Resilient In-Memory Token Bucket Fallback
	rl.memMu.Lock()
	defer rl.memMu.Unlock()

	now := time.Now()
	bucket, exists := rl.memBuckets[key]
	if !exists {
		bucket = &MemoryBucket{
			tokens:      float64(rl.capacityBurst),
			lastUpdated: now,
		}
		rl.memBuckets[key] = bucket
	} else {
		elapsed := now.Sub(bucket.lastUpdated).Seconds()
		bucket.tokens = math.Min(float64(rl.capacityBurst), bucket.tokens+elapsed*float64(rl.rateRPS))
		bucket.lastUpdated = now
	}

	if bucket.tokens >= 1.0 {
		bucket.tokens -= 1.0
		return true, int(bucket.tokens), 0
	}

	missing := 1.0 - bucket.tokens
	retryAfter = int(math.Ceil(missing / float64(rl.rateRPS)))
	if retryAfter < 1 {
		retryAfter = 1
	}
	return false, 0, retryAfter
}

// RateLimitMiddleware creates Gin middleware for rate limiting
func (rl *RateLimiter) RateLimitMiddleware(skipPrefixes []string) gin.HandlerFunc {
	return func(c *gin.Context) {
		path := c.Request.URL.Path
		for _, prefix := range skipPrefixes {
			if path == prefix || (len(prefix) > 1 && path == prefix+"/") {
				c.Next()
				return
			}
		}

		// Identify client by Tenant/User or IP address
		identifier := c.ClientIP()
		if tenantID, exists := c.Get("tenant_id"); exists && tenantID.(string) != "" {
			identifier = fmt.Sprintf("tenant:%s", tenantID)
		} else if userID, exists := c.Get("user_id"); exists && userID.(string) != "" {
			identifier = fmt.Sprintf("user:%s", userID)
		}

		allowed, remaining, retryAfter := rl.Allow(c.Request.Context(), identifier)

		c.Header("X-RateLimit-Limit", strconv.Itoa(rl.capacityBurst))
		c.Header("X-RateLimit-Remaining", strconv.Itoa(remaining))

		if !allowed {
			c.Header("Retry-After", strconv.Itoa(retryAfter))
			c.AbortWithStatusJSON(http.StatusTooManyRequests, gin.H{
				"error":               "Too Many Requests",
				"message":             "Token bucket rate limit quota exceeded",
				"limit":               rl.rateRPS,
				"burst":               rl.capacityBurst,
				"remaining":           remaining,
				"retry_after_seconds": retryAfter,
				"trace_id":            c.GetString("trace_id"),
			})
			return
		}

		c.Next()
	}
}
