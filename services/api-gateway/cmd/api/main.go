package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/enterprise-ai-platform/api-gateway/internal/config"
	"github.com/enterprise-ai-platform/api-gateway/internal/middleware"
	"github.com/enterprise-ai-platform/api-gateway/internal/proxy"
	"github.com/gin-gonic/gin"
	"github.com/redis/go-redis/v9"
)

var startTime = time.Now()

// TokenRequest payload for issuing dev/client tokens
type TokenRequest struct {
	UserID   string   `json:"user_id"`
	Email    string   `json:"email"`
	Roles    []string `json:"roles"`
	TenantID string   `json:"tenant_id"`
}

// SetupRouter initializes the Gin engine and configures all middlewares and routes
func SetupRouter(cfg *config.Config, redisClient *redis.Client) *gin.Engine {
	if cfg.Environment == "production" {
		gin.SetMode(gin.ReleaseMode)
	}

	router := gin.New()
	router.Use(gin.Recovery())

	// 1. Core Distributed Tracing & W3C context injection
	router.Use(middleware.W3CTracingMiddleware())

	// 2. Distributed Rate Limiter with in-memory fallback
	limiter := middleware.NewRateLimiter(redisClient, cfg.RateLimitRPS, cfg.RateLimitBurst)
	publicPrefixes := []string{"/healthz", "/metrics", "/api/v1/auth/token"}
	router.Use(limiter.RateLimitMiddleware(publicPrefixes))

	// 3. JWT Authentication & Claims extraction
	router.Use(middleware.JWTAuthMiddleware(cfg.JWTSecret, cfg.AuthEnabled, publicPrefixes))

	// Health and Readiness Endpoint
	router.GET("/healthz", func(c *gin.Context) {
		redisStatus := "DOWN"
		if limiter.IsRedisHealthy() {
			redisStatus = "UP"
		}

		c.JSON(http.StatusOK, gin.H{
			"status":       "UP",
			"service":      "api-gateway",
			"uptime_sec":   int(time.Since(startTime).Seconds()),
			"redis_status": redisStatus,
			"environment":  cfg.Environment,
			"auth_enabled": cfg.AuthEnabled,
			"timestamp":    time.Now().UTC().Format(time.RFC3339),
		})
	})

	// Prometheus Metrics Endpoint
	router.GET("/metrics", func(c *gin.Context) {
		metricsPayload := fmt.Sprintf(
			"# HELP gateway_uptime_seconds Total seconds gateway has been running\n"+
				"# TYPE gateway_uptime_seconds gauge\n"+
				"gateway_uptime_seconds %d\n"+
				"# HELP gateway_rate_limit_rps Configured RPS limit per client\n"+
				"# TYPE gateway_rate_limit_rps gauge\n"+
				"gateway_rate_limit_rps %d\n",
			int(time.Since(startTime).Seconds()),
			cfg.RateLimitRPS,
		)
		c.Data(http.StatusOK, "text/plain; version=0.0.4; charset=utf-8", []byte(metricsPayload))
	})

	// Public Token Issuer (for developer evaluation, automation, and CI/CD)
	router.POST("/api/v1/auth/token", func(c *gin.Context) {
		var req TokenRequest
		if err := c.ShouldBindJSON(&req); err != nil {
			// Defaults if empty payload
			req.UserID = "staff-engineer"
			req.Email = "engineer@enterprise.ai"
			req.Roles = []string{"admin", "rag-user", "ml-user"}
			req.TenantID = "tenant-enterprise-01"
		}

		if req.UserID == "" {
			req.UserID = "developer"
		}
		if len(req.Roles) == 0 {
			req.Roles = []string{"rag-user", "ml-user"}
		}
		if req.TenantID == "" {
			req.TenantID = "default"
		}

		token, err := middleware.GenerateToken(cfg.JWTSecret, req.UserID, req.Email, req.Roles, req.TenantID, 24*time.Hour)
		if err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": "Failed to generate token", "details": err.Error()})
			return
		}

		c.JSON(http.StatusOK, gin.H{
			"access_token": token,
			"token_type":   "Bearer",
			"expires_in":   86400,
			"user_id":      req.UserID,
			"roles":        req.Roles,
			"tenant_id":    req.TenantID,
		})
	})

	// Reverse Proxies for Downstream AI & ML Microservices
	ragProxy := proxy.CreateReverseProxy(cfg.RagServiceURL, "hybrid-rag-engine")
	mlopsProxy := proxy.CreateReverseProxy(cfg.MLOpsServiceURL, "mlops-inference")

	v1 := router.Group("/api/v1")
	{
		// Documents & Knowledge Retrieval (Hybrid RAG)
		v1.Any("/documents/*path", ragProxy)

		// Real-time Model Predictions & Risk Scoring (MLOps ONNX)
		v1.Any("/predictions/*path", mlopsProxy)
	}

	return router
}

func main() {
	cfg := config.LoadConfig()

	// Initialize Redis client
	var redisClient *redis.Client
	if cfg.RedisAddr != "" {
		redisClient = redis.NewClient(&redis.Options{
			Addr:     cfg.RedisAddr,
			Password: cfg.RedisPassword,
			DB:       0,
		})
	}

	router := SetupRouter(cfg, redisClient)

	server := &http.Server{
		Addr:         ":" + cfg.Port,
		Handler:      router,
		ReadTimeout:  15 * time.Second,
		WriteTimeout: 30 * time.Second,
		IdleTimeout:  60 * time.Second,
	}

	// Run in goroutine to allow graceful shutdown
	go func() {
		log.Printf("🚀 Enterprise API Gateway listening on port %s", cfg.Port)
		log.Printf("👉 Routing /api/v1/documents/*   -> %s", cfg.RagServiceURL)
		log.Printf("👉 Routing /api/v1/predictions/* -> %s", cfg.MLOpsServiceURL)
		log.Printf("🛡️ Rate Limiting: %d RPS (Burst %d) backed by Redis (%s)", cfg.RateLimitRPS, cfg.RateLimitBurst, cfg.RedisAddr)
		log.Printf("🔒 JWT Authentication: enabled=%t", cfg.AuthEnabled)

		if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("Fatal gateway server failure: %v", err)
		}
	}()

	// Graceful shutdown on SIGINT or SIGTERM
	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit
	log.Println("🛑 Shutting down Enterprise API Gateway gracefully...")

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	if err := server.Shutdown(ctx); err != nil {
		log.Fatalf("Server forced to shutdown: %v", err)
	}

	if redisClient != nil {
		_ = redisClient.Close()
	}

	log.Println("✅ API Gateway exited cleanly")
}
