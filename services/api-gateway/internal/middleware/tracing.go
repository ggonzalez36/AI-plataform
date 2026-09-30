package middleware

import (
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"log"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
)

// GenerateRandomHex generates n bytes of cryptographically secure random bytes as hex
func GenerateRandomHex(n int) string {
	bytes := make([]byte, n)
	if _, err := rand.Read(bytes); err != nil {
		// Fallback to timestamp-seeded bytes in extreme entropy failure
		fallback := fmt.Sprintf("%016x%016x", time.Now().UnixNano(), time.Now().Unix())
		if len(fallback) > n*2 {
			return fallback[:n*2]
		}
		return fallback
	}
	return hex.EncodeToString(bytes)
}

// ParseOrGenerateTraceContext parses an existing W3C traceparent or generates a new trace root
func ParseOrGenerateTraceContext(incoming string) (traceID string, spanID string, traceparent string) {
	spanID = GenerateRandomHex(8) // 16 hex chars for child span

	if incoming != "" {
		parts := strings.Split(incoming, "-")
		if len(parts) == 4 && parts[0] == "00" && len(parts[1]) == 32 && len(parts[2]) == 16 {
			// Valid W3C traceparent: reuse trace_id, create new child span_id
			traceID = parts[1]
			flags := parts[3]
			traceparent = fmt.Sprintf("00-%s-%s-%s", traceID, spanID, flags)
			return traceID, spanID, traceparent
		}
	}

	// Generate root trace context: 16 bytes (32 hex) for trace_id, 8 bytes (16 hex) for span_id, sampled flag (01)
	traceID = GenerateRandomHex(16)
	traceparent = fmt.Sprintf("00-%s-%s-01", traceID, spanID)
	return traceID, spanID, traceparent
}

// W3CTracingMiddleware injects and propagates W3C Trace Context and X-Trace-Id
func W3CTracingMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		start := time.Now()

		incomingTraceparent := c.GetHeader("traceparent")
		traceID, spanID, outgoingTraceparent := ParseOrGenerateTraceContext(incomingTraceparent)

		// Set Context values for downstream middlewares and reverse proxy
		c.Set("trace_id", traceID)
		c.Set("span_id", spanID)
		c.Set("traceparent", outgoingTraceparent)

		// Inject response headers
		c.Header("traceparent", outgoingTraceparent)
		c.Header("X-Trace-Id", traceID)

		c.Next()

		latency := time.Since(start)
		status := c.Writer.Status()

		log.Printf("[GATEWAY] %s %s | status=%d | latency=%s | trace_id=%s | traceparent=%s",
			c.Request.Method,
			c.Request.URL.Path,
			status,
			latency,
			traceID,
			outgoingTraceparent,
		)
	}
}
