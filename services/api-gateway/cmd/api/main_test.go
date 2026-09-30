package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/enterprise-ai-platform/api-gateway/internal/config"
	"github.com/enterprise-ai-platform/api-gateway/internal/middleware"
)

func createTestConfig() *config.Config {
	return &config.Config{
		Port:            "8080",
		RagServiceURL:   "http://localhost:8001",
		MLOpsServiceURL: "http://localhost:8002",
		RedisAddr:       "", // Test with in-memory fallback
		RedisPassword:   "",
		JWTSecret:       "test-secret-key-for-unit-testing-32bytes!",
		RateLimitRPS:    10,
		RateLimitBurst:  5,
		AuthEnabled:     true,
		Environment:     "test",
	}
}

func TestHealthCheck(t *testing.T) {
	cfg := createTestConfig()
	router := SetupRouter(cfg, nil)

	req, _ := http.NewRequest(http.MethodGet, "/healthz", nil)
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d", w.Code)
	}

	var resp map[string]interface{}
	if err := json.Unmarshal(w.Body.Bytes(), &resp); err != nil {
		t.Fatalf("failed to parse JSON response: %v", err)
	}

	if resp["status"] != "UP" {
		t.Errorf("expected status 'UP', got %v", resp["status"])
	}
	if resp["service"] != "api-gateway" {
		t.Errorf("expected service 'api-gateway', got %v", resp["service"])
	}

	// Verify W3C trace header injection on public routes
	traceparent := w.Header().Get("traceparent")
	if traceparent == "" || !strings.HasPrefix(traceparent, "00-") {
		t.Errorf("expected valid W3C traceparent header, got '%s'", traceparent)
	}
}

func TestMetricsEndpoint(t *testing.T) {
	cfg := createTestConfig()
	router := SetupRouter(cfg, nil)

	req, _ := http.NewRequest(http.MethodGet, "/metrics", nil)
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200, got %d", w.Code)
	}

	body := w.Body.String()
	if !strings.Contains(body, "gateway_uptime_seconds") {
		t.Errorf("expected metrics to contain gateway_uptime_seconds, got: %s", body)
	}
}

func TestW3CTraceContextPropagation(t *testing.T) {
	cfg := createTestConfig()
	router := SetupRouter(cfg, nil)

	incomingTraceID := "4bf92f3577b34da6a3ce929d0e0e4736"
	incomingTraceparent := fmt.Sprintf("00-%s-00f067aa0ba902b7-01", incomingTraceID)

	req, _ := http.NewRequest(http.MethodGet, "/healthz", nil)
	req.Header.Set("traceparent", incomingTraceparent)
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	resTraceparent := w.Header().Get("traceparent")
	resTraceID := w.Header().Get("X-Trace-Id")

	if resTraceID != incomingTraceID {
		t.Errorf("expected preserved trace ID '%s', got '%s'", incomingTraceID, resTraceID)
	}

	if !strings.Contains(resTraceparent, incomingTraceID) {
		t.Errorf("expected outgoing traceparent to contain '%s', got '%s'", incomingTraceID, resTraceparent)
	}
}

func TestTokenGenerationAndAuthValidation(t *testing.T) {
	cfg := createTestConfig()
	router := SetupRouter(cfg, nil)

	// 1. Unauthenticated request to protected route should fail with 401
	unauthReq, _ := http.NewRequest(http.MethodPost, "/api/v1/documents/query", strings.NewReader(`{"query":"test"}`))
	unauthW := httptest.NewRecorder()
	router.ServeHTTP(unauthW, unauthReq)

	if unauthW.Code != http.StatusUnauthorized {
		t.Errorf("expected 401 for unauthenticated request, got %d", unauthW.Code)
	}

	// 2. Issue valid token
	tokenReqBody := `{"user_id":"test-user","roles":["rag-user"],"tenant_id":"tenant-test"}`
	tokenReq, _ := http.NewRequest(http.MethodPost, "/api/v1/auth/token", strings.NewReader(tokenReqBody))
	tokenReq.Header.Set("Content-Type", "application/json")
	tokenW := httptest.NewRecorder()
	router.ServeHTTP(tokenW, tokenReq)

	if tokenW.Code != http.StatusOK {
		t.Fatalf("failed to generate token: %d %s", tokenW.Code, tokenW.Body.String())
	}

	var tokenResp map[string]interface{}
	_ = json.Unmarshal(tokenW.Body.Bytes(), &tokenResp)
	token, ok := tokenResp["access_token"].(string)
	if !ok || token == "" {
		t.Fatalf("expected valid access_token in response")
	}

	// 3. Test invalid Bearer token
	invalidReq, _ := http.NewRequest(http.MethodPost, "/api/v1/documents/query", strings.NewReader(`{"query":"test"}`))
	invalidReq.Header.Set("Authorization", "Bearer invalid-token-string")
	invalidW := httptest.NewRecorder()
	router.ServeHTTP(invalidW, invalidReq)

	if invalidW.Code != http.StatusUnauthorized {
		t.Errorf("expected 401 for invalid token, got %d", invalidW.Code)
	}
}

func TestRateLimiterQuota(t *testing.T) {
	cfg := createTestConfig()
	cfg.RateLimitBurst = 3 // small burst for fast testing
	cfg.RateLimitRPS = 1
	cfg.AuthEnabled = false // isolate rate limit test

	router := SetupRouter(cfg, nil)

	// First 3 requests should succeed (burst capacity = 3)
	for i := 0; i < 3; i++ {
		req, _ := http.NewRequest(http.MethodGet, "/healthz", nil)
		// Healthz is in public prefixes, so let's use a route that is rate limited
	}

	// Test directly against RateLimiter Allow method
	limiter := middleware.NewRateLimiter(nil, 1, 2)

	allowed1, rem1, _ := limiter.Allow(nil, "client-test")
	allowed2, rem2, _ := limiter.Allow(nil, "client-test")
	allowed3, _, retryAfter := limiter.Allow(nil, "client-test")

	if !allowed1 || !allowed2 {
		t.Errorf("expected first two requests to be allowed under burst=2, got %t and %t", allowed1, allowed2)
	}
	if rem1 != 1 || rem2 != 0 {
		t.Errorf("expected remaining tokens to decrease (1, 0), got %d, %d", rem1, rem2)
	}
	if allowed3 {
		t.Errorf("expected 3rd request to be rejected, but was allowed")
	}
	if retryAfter < 1 {
		t.Errorf("expected retryAfter >= 1s, got %d", retryAfter)
	}
}

func TestReverseProxyContextPropagation(t *testing.T) {
	var receivedTraceID string
	var receivedTraceparent string
	var receivedUserID string
	var receivedTenantID string

	// Create mock downstream RAG microservice
	mockDownstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		receivedTraceID = r.Header.Get("X-Trace-Id")
		receivedTraceparent = r.Header.Get("traceparent")
		receivedUserID = r.Header.Get("X-User-Id")
		receivedTenantID = r.Header.Get("X-Tenant-Id")

		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{"status":"mock_rag_response"}`))
	}))
	defer mockDownstream.Close()

	cfg := createTestConfig()
	cfg.RagServiceURL = mockDownstream.URL
	router := SetupRouter(cfg, nil)

	// Generate token for authentication
	token, err := middleware.GenerateToken(cfg.JWTSecret, "user-42", "user42@enterprise.com", []string{"rag-user"}, "tenant-99", time.Hour)
	if err != nil {
		t.Fatalf("failed to generate token: %v", err)
	}

	req, _ := http.NewRequest(http.MethodPost, "/api/v1/documents/query", strings.NewReader(`{"query":"compliance"}`))
	req.Header.Set("Authorization", "Bearer "+token)
	req.Header.Set("Content-Type", "application/json")

	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200 OK from downstream proxy, got %d. Body: %s", w.Code, w.Body.String())
	}

	if receivedTraceID == "" {
		t.Errorf("downstream did not receive X-Trace-Id")
	}
	if !strings.HasPrefix(receivedTraceparent, "00-") {
		t.Errorf("downstream did not receive valid W3C traceparent, got: %s", receivedTraceparent)
	}
	if receivedUserID != "user-42" {
		t.Errorf("expected downstream to receive X-User-Id='user-42', got '%s'", receivedUserID)
	}
	if receivedTenantID != "tenant-99" {
		t.Errorf("expected downstream to receive X-Tenant-Id='tenant-99', got '%s'", receivedTenantID)
	}
}

func TestSecurityHeaders(t *testing.T) {
	cfg := createTestConfig()
	router := SetupRouter(cfg, nil)

	req, _ := http.NewRequest(http.MethodGet, "/healthz", nil)
	w := httptest.NewRecorder()
	router.ServeHTTP(w, req)

	if w.Header().Get("X-Frame-Options") != "DENY" {
		t.Errorf("expected X-Frame-Options: DENY, got %s", w.Header().Get("X-Frame-Options"))
	}
	if w.Header().Get("X-Content-Type-Options") != "nosniff" {
		t.Errorf("expected X-Content-Type-Options: nosniff, got %s", w.Header().Get("X-Content-Type-Options"))
	}
	if !strings.Contains(w.Header().Get("Strict-Transport-Security"), "max-age=") {
		t.Errorf("expected HSTS header, got %s", w.Header().Get("Strict-Transport-Security"))
	}
	if !strings.Contains(w.Header().Get("Content-Security-Policy"), "default-src") {
		t.Errorf("expected CSP header, got %s", w.Header().Get("Content-Security-Policy"))
	}
}

func TestWAFProtectionRules(t *testing.T) {
	cfg := createTestConfig()
	router := SetupRouter(cfg, nil)

	// 1. Path traversal attack
	req1, _ := http.NewRequest(http.MethodGet, "/healthz?file=../../etc/passwd", nil)
	w1 := httptest.NewRecorder()
	router.ServeHTTP(w1, req1)
	if w1.Code != http.StatusForbidden {
		t.Errorf("expected 403 Forbidden for path traversal, got %d", w1.Code)
	}

	// 2. Cross-Site Scripting (XSS) attack
	req2, _ := http.NewRequest(http.MethodGet, "/healthz?q=<script>alert('xss')</script>", nil)
	w2 := httptest.NewRecorder()
	router.ServeHTTP(w2, req2)
	if w2.Code != http.StatusForbidden {
		t.Errorf("expected 403 Forbidden for XSS attack, got %d", w2.Code)
	}

	// 3. Malicious scanner User-Agent
	req3, _ := http.NewRequest(http.MethodGet, "/healthz", nil)
	req3.Header.Set("User-Agent", "sqlmap/1.5.2#stable")
	w3 := httptest.NewRecorder()
	router.ServeHTTP(w3, req3)
	if w3.Code != http.StatusForbidden {
		t.Errorf("expected 403 Forbidden for sqlmap scanner, got %d", w3.Code)
	}
}

