package middleware

import (
	"log"
	"net/http"
	"net/url"
	"regexp"
	"strings"

	"github.com/gin-gonic/gin"
)

// WAF Attack Signatures
var (
	pathTraversalRegex  = regexp.MustCompile(`(?i)(\.\./|\.\.\\|%2e%2e%2f|%2e%2e/|%252e%252e)`)
	commandInjectRegex  = regexp.MustCompile(`(?i)(;\s*(cat|rm|ls|wget|curl|chmod|bash|sh|powershell)\b|\|\s*(bash|sh)|` + "`" + `|\$\(.*\))`)
	xssInjectionRegex   = regexp.MustCompile(`(?i)(<script\b|javascript:|onerror\s*=|onload\s*=|alert\(|<iframe\b)`)
	sqliInjectionRegex  = regexp.MustCompile(`(?i)(\bunion\s+select\b|\bexec\s*\(|--\s*$|;\s*drop\s+table\b|'\s*or\s*'1'\s*=\s*'1)`)
	nosqliPatternRegex  = regexp.MustCompile(`(?i)(\$gt|\$ne|\$where|\$regex)`)
)

// SecurityHeadersMiddleware applies hardened OWASP HTTP response headers
func SecurityHeadersMiddleware() gin.HandlerFunc {
	return func(c *gin.Context) {
		// Strict Transport Security (HSTS)
		c.Header("Strict-Transport-Security", "max-age=31536000; includeSubDomains; preload")
		// Prevent MIME-sniffing
		c.Header("X-Content-Type-Options", "nosniff")
		// Prevent Clickjacking (framing)
		c.Header("X-Frame-Options", "DENY")
		// Cross-Site Scripting (XSS) legacy filter
		c.Header("X-XSS-Protection", "1; mode=block")
		// Content Security Policy
		c.Header("Content-Security-Policy", "default-src 'self'; frame-ancestors 'none'")
		// Privacy & Referrer
		c.Header("Referrer-Policy", "strict-origin-when-cross-origin")
		// Permissions Policy
		c.Header("Permissions-Policy", "camera=(), microphone=(), geolocation=()")

		c.Next()
	}
}

// WAFMiddleware inspects incoming requests against OWASP API Top 10 attack vectors
func WAFMiddleware(maxBodyBytes int64) gin.HandlerFunc {
	return func(c *gin.Context) {
		traceID := c.GetString("trace_id")
		clientIP := c.ClientIP()

		// 1. Anti-DoS Body Size Limitation
		if maxBodyBytes > 0 && c.Request.Body != nil {
			c.Request.Body = http.MaxBytesReader(c.Writer, c.Request.Body, maxBodyBytes)
		}

		// 2. Decode and analyze URL path and raw queries
		rawURL := c.Request.URL.Path
		if c.Request.URL.RawQuery != "" {
			rawURL += "?" + c.Request.URL.RawQuery
		}
		decodedURL, err := url.QueryUnescape(rawURL)
		if err != nil {
			decodedURL = rawURL
		}

		// Check attack vectors
		var violationRule string
		if pathTraversalRegex.MatchString(decodedURL) {
			violationRule = "RULE_PATH_TRAVERSAL"
		} else if commandInjectRegex.MatchString(decodedURL) {
			violationRule = "RULE_COMMAND_INJECTION"
		} else if xssInjectionRegex.MatchString(decodedURL) {
			violationRule = "RULE_CROSS_SITE_SCRIPTING"
		} else if sqliInjectionRegex.MatchString(decodedURL) {
			violationRule = "RULE_SQL_INJECTION"
		} else if nosqliPatternRegex.MatchString(decodedURL) {
			violationRule = "RULE_NOSQL_INJECTION"
		}

		// Check User-Agent and suspicious headers
		userAgent := c.GetHeader("User-Agent")
		if strings.Contains(strings.ToLower(userAgent), "sqlmap") || strings.Contains(strings.ToLower(userAgent), "nikto") {
			violationRule = "RULE_MALICIOUS_SCANNER_USER_AGENT"
		}

		if violationRule != "" {
			log.Printf("🚨 [SECURITY_AUDIT] event=WAF_BLOCKED rule=%s client_ip=%s path=%s trace_id=%s",
				violationRule, clientIP, c.Request.URL.Path, traceID)

			c.AbortWithStatusJSON(http.StatusForbidden, gin.H{
				"error":          "Forbidden",
				"message":        "Request blocked by Web Application Firewall (WAF) policy",
				"violation_rule": violationRule,
				"trace_id":       traceID,
			})
			return
		}

		c.Next()
	}
}
