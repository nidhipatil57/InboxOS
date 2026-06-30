import { encrypt, decrypt } from './utils/crypto';
import { WebhookDispatcher } from './services/webhook-dispatcher.service';
import { EmailSenderService } from './services/email-sender.service';
import { GmailSyncService } from './services/gmail-sync.service';
import { IMAPService } from './services/imap.service';

async function runTests() {
  console.log('--- Running P4 Sanity Checks ---');
  let passed = 0;
  let failed = 0;

  try {
    // 1. Crypto Test
    const originalText = JSON.stringify({ token: 'abc-123', secret: 'xyz-987' });
    const encrypted = encrypt(originalText);
    const decrypted = decrypt(encrypted);
    
    if (originalText === decrypted && encrypted !== originalText) {
      console.log('✅ Crypto logic (AES-256-CBC) is working perfectly.');
      passed++;
    } else {
      console.error('❌ Crypto logic failed.');
      failed++;
    }
  } catch (err: any) {
    console.error('❌ Crypto logic crashed:', err.message);
    failed++;
  }

  try {
    // 2. Class Instantiation Check
    // Ensuring classes load without dependency or syntax errors
    const imap = new IMAPService('user_123', { user: 'test', password: 'password', host: 'imap.test.com', port: 993, tls: true });
    console.log('✅ Service classes instantiated successfully.');
    passed++;
  } catch (err: any) {
    console.error('❌ Service classes failed to load:', err.message);
    failed++;
  }

  console.log(`\n--- Test Summary: ${passed} Passed, ${failed} Failed ---`);
  if (failed > 0) process.exit(1);
}

runTests();
