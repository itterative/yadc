import { storageStore } from './storageStore';

export const sessionPassword = storageStore<string | null>(
    'yadc_session_password',
    null,
    'session'
);
