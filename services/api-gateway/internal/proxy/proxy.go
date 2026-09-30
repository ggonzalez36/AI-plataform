package proxy

import (
	"encoding/json"
	"fmt"
	"log"
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"
	"time"

	"github.com/gin-gonic/gin"
)

// CreateReverseProxy returns a Gin handler configured with custom director, error handling, and telemetry headers
func CreateReverseProxy(target string, serviceName string) gin.HandlerFunc {
	targetURL, err := url.Parse(target)
	if err != nil {
		log.Fatalf("Invalid proxy target URL for service %s: %v", serviceName, err)
	}

	proxy := httputil.NewSingleHostReverseProxy(targetURL)

	// Custom optimized transport with connection pooling
	proxy.Transport = &http.Transport{
		Proxy: http.ProxyFromEnvironment,
		DialContext: (&net.Dialer{
			Timeout:   5 * time.Second,
			KeepAlive: 30 * time.Second,
		}).DialContext,
		MaxIdleConns:          100,
		MaxIdleConnsPerHost:   50,
		IdleConnTimeout:       90 * time.Second,
		TLSHandshakeTimeout:   5 * time.Second,
		ResponseHeaderTimeout: 30 * time.Second,
	}

	// Custom director propagating distributed tracing & identity headers
	originalDirector := proxy.Director
	proxy.Director = func(req *http.Request) {
		originalDirector(req)
		req.Host = targetURL.Host
		req.Header.Set("X-Forwarded-Host", req.Host)
		req.Header.Set("X-Forwarded-Proto", targetURL.Scheme)
	}

	// Resilient error handler for downstream unavailability
	proxy.ErrorHandler = func(w http.ResponseWriter, req *http.Request, proxyErr error) {
		traceID := req.Header.Get("X-Trace-Id")
		log.Printf("❌ Downstream proxy error [%s]: %v (trace_id=%s)", serviceName, proxyErr, traceID)

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusBadGateway)
		_ = json.NewEncoder(w).Encode(map[string]interface{}{
			"error":       "Bad Gateway",
			"message":     fmt.Sprintf("Downstream service '%s' is unreachable or timed out", serviceName),
			"service":     serviceName,
			"target":      target,
			"details":     proxyErr.Error(),
			"trace_id":    traceID,
			"status_code": http.StatusBadGateway,
		})
	}

	return func(c *gin.Context) {
		// Context propagation to downstream headers
		if traceparent := c.GetString("traceparent"); traceparent != "" {
			c.Request.Header.Set("traceparent", traceparent)
		}
		if traceID := c.GetString("trace_id"); traceID != "" {
			c.Request.Header.Set("X-Trace-Id", traceID)
		}
		if userID := c.GetString("user_id"); userID != "" {
			c.Request.Header.Set("X-User-Id", userID)
		}
		if tenantID := c.GetString("tenant_id"); tenantID != "" {
			c.Request.Header.Set("X-Tenant-Id", tenantID)
		}
		if roles := c.GetString("roles"); roles != "" {
			c.Request.Header.Set("X-User-Roles", roles)
		}

		proxy.ServeHTTP(c.Writer, c.Request)
	}
}
