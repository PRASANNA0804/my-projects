/**
 * auth-bridge.js — ARIA Authentication Gatekeeper
 * ================================================
 * HOW TO CONNECT:   Add this as the FIRST <script> tag in index_v2.html <head>:
 *                   <script src="auth-bridge.js"></script>
 *
 * HOW TO DISCONNECT: Remove that single <script> line. Done.
 *
 * This script checks for a valid login session.
 * If none found, user is redirected to login.html immediately.
 */
(function () {
  'use strict';
  const SESSION_KEY = 'aria_auth';
  const LOGIN_PAGE  = 'login.html';
  const SESSION_TTL = 8 * 60 * 60 * 1000; // 8 hours

  try {
    const raw  = sessionStorage.getItem(SESSION_KEY);
    if (!raw) { window.location.replace(LOGIN_PAGE); return; }
    const data = JSON.parse(raw);
    if (!data || !data.time || (Date.now() - data.time) > SESSION_TTL) {
      sessionStorage.removeItem(SESSION_KEY);
      window.location.replace(LOGIN_PAGE);
    }
  } catch (e) {
    window.location.replace(LOGIN_PAGE);
  }
})();
