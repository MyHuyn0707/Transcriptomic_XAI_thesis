import React from 'react';

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
        <div className="min-h-screen flex items-center justify-center bg-slate-50 p-8">
          <div className="max-w-md text-center bg-white border border-rose-200 rounded-2xl shadow-sm p-8">
            <h1 className="text-lg font-bold text-rose-600 mb-2">Đã xảy ra lỗi</h1>
            <p className="text-sm text-slate-500 mb-4">
              Giao diện gặp lỗi không mong muốn. Vui lòng tải lại trang.
            </p>
            <p className="text-xs font-mono text-slate-400 break-words">{this.state.error.message}</p>
            <button
              onClick={() => window.location.reload()}
              className="mt-5 bg-teal-600 hover:bg-teal-700 text-white font-medium py-2 px-4 rounded-lg text-sm"
            >
              Tải lại trang
            </button>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}
