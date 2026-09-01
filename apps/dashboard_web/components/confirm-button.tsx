'use client';

import { useEffect, useRef, useState } from 'react';

export interface ConfirmButtonProps {
  label: string;
  confirmLabel?: string;
  onConfirm: () => void;
  pending?: boolean;
  disabled?: boolean;
  variant?: 'default' | 'danger';
}

const ARM_RESET_MS = 4000;

/**
 * Two-step inline confirmation for destructive workflow actions (Stop/Kill).
 * First click arms; the armed state offers confirm/cancel and auto-resets
 * after 4 seconds so a stale confirm never lingers.
 *
 * Keyboard support: focus follows the state swap (arm -> Confirm button,
 * cancel/auto-reset -> primary trigger), and the auto-reset timer is
 * suspended while focus stays inside the group so a keyboard user is never
 * disarmed mid-decision.
 */
export function ConfirmButton({
  label,
  confirmLabel,
  onConfirm,
  pending = false,
  disabled = false,
  variant = 'default',
}: ConfirmButtonProps) {
  const [armed, setArmed] = useState(false);
  const resetTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const primaryRef = useRef<HTMLButtonElement | null>(null);
  const confirmRef = useRef<HTMLButtonElement | null>(null);
  const groupRef = useRef<HTMLSpanElement | null>(null);
  // Focus is moved in an effect (the target button mounts on the same render
  // that flips `armed`); track where focus should land after the swap.
  const focusTargetRef = useRef<'confirm' | 'primary' | null>(null);

  const clearResetTimer = () => {
    if (resetTimerRef.current !== null) {
      clearTimeout(resetTimerRef.current);
      resetTimerRef.current = null;
    }
  };

  useEffect(() => clearResetTimer, []);

  useEffect(() => {
    if (focusTargetRef.current === 'confirm' && armed) {
      confirmRef.current?.focus();
      focusTargetRef.current = null;
    } else if (focusTargetRef.current === 'primary' && !armed) {
      primaryRef.current?.focus();
      focusTargetRef.current = null;
    }
  });

  const startResetTimer = () => {
    clearResetTimer();
    resetTimerRef.current = setTimeout(() => setArmed(false), ARM_RESET_MS);
  };

  const disarm = (returnFocus: boolean) => {
    clearResetTimer();
    if (returnFocus) {
      focusTargetRef.current = 'primary';
    }
    setArmed(false);
  };

  const arm = () => {
    focusTargetRef.current = 'confirm';
    setArmed(true);
    startResetTimer();
  };

  const handleGroupFocus = () => {
    // Never auto-disarm under the keyboard user's focus.
    clearResetTimer();
  };

  const handleGroupBlur = (event: React.FocusEvent<HTMLSpanElement>) => {
    if (!armed) {
      return;
    }
    const next = event.relatedTarget as Node | null;
    if (next === null || !groupRef.current?.contains(next)) {
      startResetTimer();
    }
  };

  const buttonClass = `action-button confirm-button${variant === 'danger' ? ' confirm-button-danger' : ''}`;
  const busy = pending || disabled;

  return (
    <span
      ref={groupRef}
      className="confirm-button-group"
      role="group"
      aria-label={`Confirm ${label}`}
      onFocus={handleGroupFocus}
      onBlur={handleGroupBlur}
    >
      {!armed ? (
        <button
          ref={primaryRef}
          type="button"
          className={buttonClass}
          onClick={arm}
          disabled={busy}
          aria-busy={pending}
        >
          {pending ? `${label}…` : label}
        </button>
      ) : (
        <>
          <button
            ref={confirmRef}
            type="button"
            className={buttonClass}
            onClick={() => {
              disarm(true);
              onConfirm();
            }}
            disabled={busy}
            aria-busy={pending}
          >
            {confirmLabel ?? `Confirm ${label.toLowerCase()}?`}
          </button>
          <button
            type="button"
            className="action-button confirm-button-cancel"
            onClick={() => disarm(true)}
          >
            Cancel
          </button>
        </>
      )}
    </span>
  );
}
