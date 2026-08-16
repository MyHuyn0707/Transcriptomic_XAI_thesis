import React from 'react';
import Button from './ui/Button';

interface Props {
  children: React.ReactNode;
}

interface State {
  error: Error | null;
}

export default class ErrorBoundary extends React.Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: React.ErrorInfo) {
    console.error('Unhandled UI error:', error, info.componentStack);
  }

  render() {
    if (this.state.error) {
      return (
        <div className="min-h-screen flex items-center justify-center bg-neutral-50 p-8">
          <div className="max-w-md text-center bg-white border border-danger-200 rounded-2xl shadow-sm p-8">
            <h1 className="text-lg font-bold text-danger-600 mb-2">Đã xảy ra lỗi</h1>
            <p className="text-sm text-neutral-600 mb-4">
              Giao diện gặp lỗi không mong muốn. Vui lòng tải lại trang.
            </p>
            <p className="text-xs font-mono text-neutral-500 break-words">{this.state.error.message}</p>
            <Button onClick={() => window.location.reload()} className="mt-5 px-4">
              Tải lại trang
            </Button>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}
